//
//  NFKServedAnthropic.m
//  InferKit
//

#import "NFKServedAnthropic.h"
#import "NFKServedOpenAI.h"
#import "NFKInferenceWireCoding.h"
#import "NFKInferenceServerPrivate.h"
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKInferenceTool.h>
#import <InferKit/NFKToolMessages.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKErrors.h>

/*! The contract effort for each effort level the API names; any other level passes through. */
static NSString *NFKServedAnthropicEffort(id level)
{
	NSDictionary<NSString *, NSString *> *efforts = @{ @"low": NFKReasoningEffortLight,
													   @"medium": NFKReasoningEffortModerate,
													   @"high": NFKReasoningEffortDeep };
	return efforts[level] ?: level;
}

static NSString *NFKServedAnthropicJSONText(id object)
{
	if (object == nil || ![NSJSONSerialization isValidJSONObject:object]) {
		return @"{}";
	}
	NSData *data = [NSJSONSerialization dataWithJSONObject:object options:0 error:NULL];
	return data != nil ? [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] : @"{}";
}

static NSString *NFKServedAnthropicIdentifier(NSString *prefix)
{
	NSString *digits = [NSUUID.UUID.UUIDString stringByReplacingOccurrencesOfString:@"-" withString:@""].lowercaseString;
	return [prefix stringByAppendingString:digits];
}

@implementation NFKServedAnthropic

#pragma mark Routes

+ (void)serveMessages:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response server:(NFKInferenceServer *)server
{
	NSDictionary *body = [server JSONBodyOf:request response:response wire:NFKServedWireAnthropic];
	if (body == nil) {
		return;
	}
	NSError *error = nil;
	NSString *forcedToolName = nil;
	NFKServedModel *model = [server modelNamed:body[@"model"] error:&error];
	NFKInferenceRequest *inference = model != nil ? [self requestFromBody:body forcedToolName:&forcedToolName error:&error] : nil;
	if (inference == nil) {
		[server sendError:error response:response wire:NFKServedWireAnthropic];
		return;
	}
	NSString *identifier = [self newMessageIdentifier];
	NSString *modelName = NFKServedModelName(model);
	NFKServedRun *run = [[NFKServedRun alloc] init];
	run.request = inference;
	if (![body[@"stream"] boolValue]) {
		run.completionHandler = ^(NFKInferenceResult *result, NSError *runError) {
			if (result == nil) {
				[server sendError:runError response:response wire:NFKServedWireAnthropic];
				return;
			}
			[response sendStatus:200 JSONObject:[self messageForResult:result request:inference identifier:identifier
																  model:modelName forcedToolName:forcedToolName] headers:nil];
		};
		[server enqueue:run on:model request:request response:response];
		return;
	}
	NFKServedAnthropicStream *stream = [[NFKServedAnthropicStream alloc] initWithRequest:inference identifier:identifier model:modelName
																		  forcedToolName:forcedToolName eventHandler:NFKServedEventWriter(response)];
	NSObject *lock = [[NSObject alloc] init];
	run.partialHandler = ^(NFKInferenceResult *partial, double progress) {
		@synchronized (lock) {
			if (partial != nil) {
				[stream appendPartial:partial];
			}
		}
	};
	run.completionHandler = ^(NFKInferenceResult *result, NSError *runError) {
		@synchronized (lock) {
			if (result == nil && !stream.started) {
				[server sendError:runError response:response wire:NFKServedWireAnthropic];
				return;
			}
			if (result != nil) {
				[stream finishWithResult:result];
			} else {
				NSError *failure = runError ?: [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceBackendFailure
															   userInfo:@{ NSLocalizedDescriptionKey: @"the run failed" }];
				[stream failWithError:failure status:[server statusForError:failure]];
			}
			[response endStream];
		}
	};
	[server enqueue:run on:model request:request response:response];
}

+ (void)serveTokenCount:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response server:(NFKInferenceServer *)server
{
	NSDictionary *body = [server JSONBodyOf:request response:response wire:NFKServedWireAnthropic];
	if (body == nil) {
		return;
	}
	NSError *error = nil;
	NFKServedModel *model = [server modelNamed:body[@"model"] error:&error];
	NFKInferenceRequest *inference = model != nil ? [self requestFromBody:body forcedToolName:NULL error:&error] : nil;
	if (inference == nil) {
		[server sendError:error response:response wire:NFKServedWireAnthropic];
		return;
	}
	NFKServedRun *run = [NFKServedRun tokenCountRunForRequest:inference modelName:NFKServedModelName(model)];
	run.completionHandler = ^(NFKInferenceResult *result, NSError *runError) {
		NSNumber *count = result != nil ? [result outputForKey:NFKOutputUsage][NFKUsageInputTokens] : nil;
		if (count == nil) {
			[server sendError:runError response:response wire:NFKServedWireAnthropic];
			return;
		}
		[response sendStatus:200 JSONObject:@{ @"input_tokens": count } headers:nil];
	};
	[server enqueue:run on:model request:request response:response];
}

+ (NSDictionary<NSString *, id> *)modelEntryForName:(NSString *)name createdAt:(NSInteger)createdAt
{
	NSISO8601DateFormatter *formatter = [[NSISO8601DateFormatter alloc] init];
	NSString *created = [formatter stringFromDate:[NSDate dateWithTimeIntervalSince1970:createdAt]];
	return @{ @"type": @"model", @"id": name, @"display_name": name, @"created_at": created };
}

+ (NSDictionary<NSString *, id> *)modelListForNames:(NSArray<NSString *> *)names createdAt:(NSInteger)createdAt
{
	NSMutableArray<NSDictionary *> *data = [NSMutableArray arrayWithCapacity:names.count];
	for (NSString *name in names) {
		[data addObject:[self modelEntryForName:name createdAt:createdAt]];
	}
	return @{ @"data": data, @"has_more": @NO, @"first_id": names.firstObject ?: NSNull.null, @"last_id": names.lastObject ?: NSNull.null };
}

#pragma mark Request

+ (nullable NFKInferenceRequest *)requestFromBody:(NSDictionary *)body
								   forcedToolName:(NSString * _Nullable * _Nullable)outForcedToolName
											error:(NSError * _Nullable *)outError
{
	NSArray *wireMessages = [body[@"messages"] isKindOfClass:NSArray.class] ? body[@"messages"] : nil;
	if (wireMessages.count == 0) {
		[NFKServedOpenAI setError:outError code:kNFKError_InferenceMissingInput reason:@"the request carries no messages"];
		return nil;
	}
	NSMutableArray<NSDictionary *> *messages = [NSMutableArray array];
	NSString *system = [self systemTextOf:body[@"system"]];
	if (system.length > 0) {
		[messages addObject:@{ @"role": @"system", @"content": system }];
	}
	NSMutableArray *images = [NSMutableArray array];
	NSMutableArray *documents = [NSMutableArray array];
	NSMutableDictionary<NSString *, NSString *> *toolNames = [NSMutableDictionary dictionary];
	for (NSDictionary *message in wireMessages) {
		if (![self readMessage:message into:messages images:images documents:documents toolNames:toolNames error:outError]) {
			return nil;
		}
	}
	NSString *forced = nil;
	NSDictionary *parameters = [self parametersFromBody:body forcedToolName:&forced error:outError];
	if (parameters == nil) {
		return nil;
	}
	if (outForcedToolName != NULL) {
		*outForcedToolName = forced;
	}

	NSMutableDictionary<NSString *, id> *inputs = [NSMutableDictionary dictionaryWithObject:messages forKey:NFKInputMessages];
	NSDictionary *only = messages.count == 1 ? messages.firstObject : nil;
	if ([only[@"role"] isEqual:@"user"] && [only[@"content"] isKindOfClass:NSString.class]) {
		inputs[NFKInputPrompt] = only[@"content"];
	}
	[self addMedia:images firstKey:NFKInputImage restKey:NFKInputImages to:inputs];
	[self addMedia:documents firstKey:NFKInputDocument restKey:NFKInputDocuments to:inputs];
	return [NFKInferenceRequest requestWithInputs:inputs parameters:parameters outputModality:NFKModalityText];
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

/*! The system field as text: a string, or its text blocks joined by a blank line. */
+ (nullable NSString *)systemTextOf:(nullable id)system
{
	if ([system isKindOfClass:NSString.class]) {
		return system;
	}
	NSMutableArray<NSString *> *texts = [NSMutableArray array];
	for (NSDictionary *block in [system isKindOfClass:NSArray.class] ? system : @[]) {
		if ([block isKindOfClass:NSDictionary.class] && [block[@"text"] isKindOfClass:NSString.class]) {
			[texts addObject:block[@"text"]];
		}
	}
	return [texts componentsJoinedByString:@"\n\n"];
}

// A user turn's tool_result blocks become tool turns ahead of the turn's own text, the order the API
// requires them in; an assistant turn's tool_use blocks become its tool_calls. Thinking blocks carry
// a signature only Anthropic can verify, so they are not replayed to the hosted backend.
+ (BOOL)readMessage:(NSDictionary *)message
			   into:(NSMutableArray<NSDictionary *> *)messages
			 images:(NSMutableArray *)images
		  documents:(NSMutableArray *)documents
		  toolNames:(NSMutableDictionary<NSString *, NSString *> *)toolNames
			  error:(NSError * _Nullable *)outError
{
	NSString *role = [message isKindOfClass:NSDictionary.class] && [message[@"role"] isKindOfClass:NSString.class] ? message[@"role"] : nil;
	if (![role isEqualToString:@"user"] && ![role isEqualToString:@"assistant"]) {
		return [NFKServedOpenAI setError:outError code:kNFKError_InferenceMissingInput
								  reason:@"a message must be an object whose role is user or assistant"];
	}
	id content = message[@"content"];
	if ([content isKindOfClass:NSString.class]) {
		[messages addObject:@{ @"role": role, @"content": content }];
		return YES;
	}
	if (![content isKindOfClass:NSArray.class]) {
		return [NFKServedOpenAI setError:outError code:kNFKError_InferenceMissingInput reason:@"a message's content is neither text nor blocks"];
	}
	NSMutableArray<NSString *> *texts = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *calls = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *toolTurns = [NSMutableArray array];
	for (NSDictionary *block in content) {
		NSString *type = [block isKindOfClass:NSDictionary.class] && [block[@"type"] isKindOfClass:NSString.class] ? block[@"type"] : @"";
		if ([type isEqualToString:@"text"]) {
			[texts addObject:[block[@"text"] isKindOfClass:NSString.class] ? block[@"text"] : @""];
		} else if ([type isEqualToString:@"image"]) {
			id image = [self imageForSource:block[@"source"] error:outError];
			if (image == nil) {
				return NO;
			}
			[images addObject:image];
		} else if ([type isEqualToString:@"document"]) {
			id document = [self documentForSource:block[@"source"] error:outError];
			if (document == nil) {
				return NO;
			}
			[documents addObject:document];
		} else if ([type isEqualToString:@"tool_use"]) {
			NSString *identifier = [block[@"id"] isKindOfClass:NSString.class] ? block[@"id"] : @"";
			NSString *name = [block[@"name"] isKindOfClass:NSString.class] ? block[@"name"] : @"";
			NSDictionary *input = [block[@"input"] isKindOfClass:NSDictionary.class] ? block[@"input"] : @{};
			toolNames[identifier] = name;
			[calls addObject:@{ @"id": identifier, @"name": name, @"arguments": input, @"argumentsJSON": NFKServedAnthropicJSONText(input) }];
		} else if ([type isEqualToString:@"tool_result"]) {
			NSDictionary *turn = [self toolTurnForBlock:block toolNames:toolNames error:outError];
			if (turn == nil) {
				return NO;
			}
			[toolTurns addObject:turn];
		} else if (![type isEqualToString:@"thinking"] && ![type isEqualToString:@"redacted_thinking"]) {
			return [NFKServedOpenAI setError:outError code:kNFKError_InferenceUnsupported
									  reason:[NSString stringWithFormat:@"the content block type \"%@\" is not served", type]];
		}
	}
	NSString *text = [texts componentsJoinedByString:@"\n"];
	if ([role isEqualToString:@"assistant"]) {
		[messages addObject:[NFKToolMessages assistantMessageWithText:text toolCalls:calls record:nil format:nil source:nil]];
		return YES;
	}
	[messages addObjectsFromArray:toolTurns];
	if (texts.count > 0 || toolTurns.count == 0) {
		[messages addObject:@{ @"role": @"user", @"content": text }];
	}
	return YES;
}

+ (nullable NSDictionary *)toolTurnForBlock:(NSDictionary *)block
								  toolNames:(NSDictionary<NSString *, NSString *> *)toolNames
									  error:(NSError * _Nullable *)outError
{
	NSString *identifier = [block[@"tool_use_id"] isKindOfClass:NSString.class] ? block[@"tool_use_id"] : nil;
	NSString *name = identifier != nil ? toolNames[identifier] : nil;
	if (name == nil) {
		[NFKServedOpenAI setError:outError code:kNFKError_InferenceMissingInput
						   reason:@"a tool_result names a tool_use_id that no earlier tool_use block carries"];
		return nil;
	}
	NSMutableArray<NSString *> *texts = [NSMutableArray array];
	NSMutableArray<NSData *> *pngs = [NSMutableArray array];
	id content = block[@"content"];
	if ([content isKindOfClass:NSString.class]) {
		[texts addObject:content];
	}
	for (NSDictionary *part in [content isKindOfClass:NSArray.class] ? content : @[]) {
		NSString *type = [part isKindOfClass:NSDictionary.class] ? part[@"type"] : nil;
		if ([type isEqual:@"text"] && [part[@"text"] isKindOfClass:NSString.class]) {
			[texts addObject:part[@"text"]];
			continue;
		}
		NSData *png = [type isEqual:@"image"] ? [self PNGDataForSource:part[@"source"] error:outError] : nil;
		if (png == nil) {
			if ([type isEqual:@"image"]) {
				return nil;
			}
			[NFKServedOpenAI setError:outError code:kNFKError_InferenceUnsupported
							   reason:[NSString stringWithFormat:@"a tool_result block of type \"%@\" is not served", type ?: @"(none)"]];
			return nil;
		}
		[pngs addObject:png];
	}
	NFKToolResult *result = [[NFKToolResult alloc] initWithText:[texts componentsJoinedByString:@"\n"] outputs:nil
													  imagePNGs:pngs isError:[block[@"is_error"] boolValue]];
	return [NFKToolMessages messageForToolCallIdentifier:identifier name:name result:result];
}

/*! The bytes of a base64 source, or nil with an error naming what the server does not serve. */
+ (nullable NSData *)bytesOfSource:(id)source error:(NSError * _Nullable *)outError
{
	NSDictionary *described = [source isKindOfClass:NSDictionary.class] ? source : nil;
	NSData *bytes = [described[@"type"] isEqual:@"base64"] && [described[@"data"] isKindOfClass:NSString.class]
		? [[NSData alloc] initWithBase64EncodedString:described[@"data"] options:NSDataBase64DecodingIgnoreUnknownCharacters] : nil;
	if (bytes == nil) {
		[NFKServedOpenAI setError:outError code:kNFKError_InferenceUnsupported
						   reason:@"media must be inline base64 data; this server fetches no URL and keeps no files"];
	}
	return bytes;
}

+ (nullable id)imageForSource:(id)source error:(NSError * _Nullable *)outError
{
	NSData *bytes = [self bytesOfSource:source error:outError];
	CVPixelBufferRef buffer = bytes != nil ? [NFKImageCoding pixelBufferWithImageData:bytes] : NULL;
	if (buffer == NULL) {
		if (bytes != nil) {
			[NFKServedOpenAI setError:outError code:kNFKError_InferenceUnsupported reason:@"an image block is not an image this server can decode"];
		}
		return nil;
	}
	return (__bridge_transfer id)buffer;
}

+ (nullable NSData *)PNGDataForSource:(id)source error:(NSError * _Nullable *)outError
{
	if ([source isKindOfClass:NSDictionary.class] && [source[@"media_type"] isEqual:@"image/png"]) {
		return [self bytesOfSource:source error:outError];
	}
	id image = [self imageForSource:source error:outError];
	return image != nil ? [NFKImageCoding PNGDataForImage:image] : nil;
}

/*! A document block's source: a PDF as its bytes, plain text as a string. */
+ (nullable id)documentForSource:(id)source error:(NSError * _Nullable *)outError
{
	if ([source isKindOfClass:NSDictionary.class] && [source[@"type"] isEqual:@"text"] && [source[@"data"] isKindOfClass:NSString.class]) {
		return source[@"data"];
	}
	return [self bytesOfSource:source error:outError];
}

#pragma mark Parameters

+ (nullable NSDictionary<NSString *, id> *)parametersFromBody:(NSDictionary *)body
											   forcedToolName:(NSString * _Nullable * _Nonnull)outForcedToolName
														error:(NSError * _Nullable *)outError
{
	NSMutableDictionary<NSString *, id> *parameters = [NSMutableDictionary dictionary];
	NSSet<NSString *> *consumed = [NSSet setWithArray:@[ @"model", @"messages", @"system", @"max_tokens", @"temperature", @"top_p",
														 @"top_k", @"stop_sequences", @"tools", @"tool_choice", @"thinking",
														 @"output_config", @"stream", @"metadata" ]];
	for (NSString *key in body) {
		if (![consumed containsObject:key]) {
			parameters[key] = body[key];
		}
	}
	NSDictionary<NSString *, NSString *> *renamed = @{ @"max_tokens": NFKParameterMaxTokens, @"temperature": NFKParameterTemperature,
													   @"top_p": NFKParameterTopP, @"top_k": NFKParameterTopK };
	for (NSString *key in renamed) {
		if ([body[key] isKindOfClass:NSNumber.class]) {
			parameters[renamed[key]] = body[key];
		}
	}
	if ([body[@"stop_sequences"] isKindOfClass:NSArray.class]) {
		parameters[NFKParameterStopSequences] = body[@"stop_sequences"];
	}
	NSMutableArray<NSDictionary *> *tools = [NSMutableArray array];
	NSMutableDictionary<NSString *, NSDictionary *> *schemas = [NSMutableDictionary dictionary];
	for (NSDictionary *tool in [body[@"tools"] isKindOfClass:NSArray.class] ? body[@"tools"] : @[]) {
		// A versioned server tool (web search, code execution) runs at Anthropic and has no schema here.
		if (![tool isKindOfClass:NSDictionary.class] || ![tool[@"name"] isKindOfClass:NSString.class]
			|| ![tool[@"input_schema"] isKindOfClass:NSDictionary.class]) {
			continue;
		}
		NSMutableDictionary *contract = [NSMutableDictionary dictionaryWithDictionary:@{ @"name": tool[@"name"], @"parameters": tool[@"input_schema"] }];
		contract[@"description"] = [tool[@"description"] isKindOfClass:NSString.class] ? tool[@"description"] : nil;
		[tools addObject:contract];
		schemas[tool[@"name"]] = tool[@"input_schema"];
	}
	if (tools.count > 0) {
		parameters[NFKParameterTools] = tools;
	}
	NSDictionary *choice = [body[@"tool_choice"] isKindOfClass:NSDictionary.class] ? body[@"tool_choice"] : nil;
	if ([choice[@"type"] isEqual:@"tool"]) {
		NSDictionary *schema = [choice[@"name"] isKindOfClass:NSString.class] ? schemas[choice[@"name"]] : nil;
		if (schema == nil) {
			[NFKServedOpenAI setError:outError code:kNFKError_InferenceMissingInput reason:@"tool_choice names a tool the request does not declare"];
			return nil;
		}
		*outForcedToolName = choice[@"name"];
		parameters[NFKParameterJSONSchema] = schema;
		[parameters removeObjectForKey:NFKParameterTools];
	} else if (choice != nil) {
		NSDictionary<NSString *, NSString *> *choices = @{ @"auto": @"auto", @"any": @"required", @"none": @"none" };
		parameters[@"tool_choice"] = choices[choice[@"type"]];
	}
	if ([choice[@"disable_parallel_tool_use"] boolValue]) {
		parameters[@"parallel_tool_calls"] = @NO;
	}
	[self addReasoningOfBody:body to:parameters];
	NSDictionary *config = [body[@"output_config"] isKindOfClass:NSDictionary.class] ? body[@"output_config"] : nil;
	NSDictionary *format = [config[@"format"] isKindOfClass:NSDictionary.class] ? config[@"format"] : nil;
	if ([format[@"type"] isEqual:@"json_schema"] && [format[@"schema"] isKindOfClass:NSDictionary.class] && *outForcedToolName == nil) {
		parameters[NFKParameterJSONSchema] = format[@"schema"];
	}
	return parameters;
}

// A thinking budget passes through as its token count in a string, which the contract's effort key
// takes. Adaptive thinking takes output_config's effort, and the API's default effort is high.
+ (void)addReasoningOfBody:(NSDictionary *)body to:(NSMutableDictionary<NSString *, id> *)parameters
{
	NSDictionary *thinking = [body[@"thinking"] isKindOfClass:NSDictionary.class] ? body[@"thinking"] : nil;
	NSDictionary *config = [body[@"output_config"] isKindOfClass:NSDictionary.class] ? body[@"output_config"] : nil;
	NSString *effort = [config[@"effort"] isKindOfClass:NSString.class] ? config[@"effort"] : nil;
	if ([thinking[@"type"] isEqual:@"disabled"]) {
		return;
	}
	if ([thinking[@"type"] isEqual:@"enabled"] && [thinking[@"budget_tokens"] isKindOfClass:NSNumber.class]) {
		parameters[NFKParameterReasoningEffort] = [thinking[@"budget_tokens"] stringValue];
	} else if (effort != nil) {
		parameters[NFKParameterReasoningEffort] = NFKServedAnthropicEffort(effort);
	} else if ([thinking[@"type"] isEqual:@"adaptive"]) {
		parameters[NFKParameterReasoningEffort] = NFKReasoningEffortDeep;
	}
}

#pragma mark Reply

+ (NSString *)newMessageIdentifier
{
	return NFKServedAnthropicIdentifier(@"msg_");
}

/*! The structured reply a forced tool carries: NFKOutputStructured, or the text when it is a JSON
	object. */
+ (NSDictionary *)forcedInputForResult:(NFKInferenceResult *)result
{
	if (result.structured != nil) {
		return result.structured;
	}
	NSData *text = [result.text dataUsingEncoding:NSUTF8StringEncoding];
	id parsed = text != nil ? [NSJSONSerialization JSONObjectWithData:text options:0 error:NULL] : nil;
	return [parsed isKindOfClass:NSDictionary.class] ? parsed : @{};
}

/*! The tool_use blocks a result calls for, each with its id, name, and input. */
+ (NSArray<NSDictionary *> *)toolUsesForResult:(NFKInferenceResult *)result forcedToolName:(nullable NSString *)forcedToolName
{
	if (forcedToolName != nil) {
		return @[ @{ @"type": @"tool_use", @"id": NFKServedAnthropicIdentifier(@"toolu_"), @"name": forcedToolName,
					 @"input": [self forcedInputForResult:result] } ];
	}
	NSMutableArray<NSDictionary *> *uses = [NSMutableArray array];
	for (NSDictionary *call in result.toolCalls ?: @[]) {
		if (![call isKindOfClass:NSDictionary.class] || ![call[@"name"] isKindOfClass:NSString.class]) {
			continue;
		}
		id arguments = call[@"arguments"];
		if (![arguments isKindOfClass:NSDictionary.class] && [call[@"argumentsJSON"] isKindOfClass:NSString.class]) {
			arguments = [NSJSONSerialization JSONObjectWithData:[call[@"argumentsJSON"] dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
		}
		NSString *identifier = [call[@"id"] isKindOfClass:NSString.class] && [call[@"id"] length] > 0 ? call[@"id"] : NFKServedAnthropicIdentifier(@"toolu_");
		[uses addObject:@{ @"type": @"tool_use", @"id": identifier, @"name": call[@"name"],
						   @"input": [arguments isKindOfClass:NSDictionary.class] ? arguments : @{} }];
	}
	return uses;
}

+ (NSString *)stopReasonForResult:(NFKInferenceResult *)result request:(NFKInferenceRequest *)request callsTools:(BOOL)callsTools
{
	if (callsTools) {
		return @"tool_use";
	}
	NSNumber *limit = [request parameterForKey:NFKParameterMaxTokens];
	NSDictionary *usage = [result outputForKey:NFKOutputUsage];
	NSNumber *produced = [usage isKindOfClass:NSDictionary.class] ? usage[NFKUsageOutputTokens] : nil;
	if ([limit isKindOfClass:NSNumber.class] && [produced isKindOfClass:NSNumber.class] && limit.integerValue > 0
		&& produced.integerValue >= limit.integerValue) {
		return @"max_tokens";
	}
	return @"end_turn";
}

/*! The usage object. Every reply carries one, with zeros for what the backend does not report. */
+ (NSDictionary<NSString *, id> *)usageForResult:(nullable NFKInferenceResult *)result
{
	NSDictionary *usage = [[result outputForKey:NFKOutputUsage] isKindOfClass:NSDictionary.class] ? [result outputForKey:NFKOutputUsage] : @{};
	NSInteger input = [usage[NFKUsageInputTokens] integerValue];
	NSNumber *cached = [usage[NFKUsageCachedTokens] isKindOfClass:NSNumber.class] ? usage[NFKUsageCachedTokens] : nil;
	NSInteger cachedCount = MIN(cached.integerValue, input);
	NSMutableDictionary *wire = [@{ @"input_tokens": @(input - cachedCount),
									@"output_tokens": @([usage[NFKUsageOutputTokens] integerValue]) } mutableCopy];
	if (cached != nil) {
		wire[@"cache_read_input_tokens"] = @(cachedCount);
	}
	return wire;
}

+ (nullable NSString *)reasoningForResult:(nullable NFKInferenceResult *)result
{
	NSString *reasoning = [result outputForKey:NFKOutputReasoning];
	return [reasoning isKindOfClass:NSString.class] ? reasoning : nil;
}

+ (NSDictionary<NSString *, id> *)messageForResult:(NFKInferenceResult *)result
										   request:(NFKInferenceRequest *)request
										identifier:(NSString *)identifier
											 model:(NSString *)model
									forcedToolName:(nullable NSString *)forcedToolName
{
	NSMutableArray<NSDictionary *> *content = [NSMutableArray array];
	NSString *reasoning = [self reasoningForResult:result];
	if (reasoning.length > 0) {
		[content addObject:@{ @"type": @"thinking", @"thinking": reasoning, @"signature": @"" }];
	}
	NSArray<NSDictionary *> *uses = [self toolUsesForResult:result forcedToolName:forcedToolName];
	NSString *text = forcedToolName != nil ? nil : [NFKServedOpenAI textForResult:result];
	if (text.length > 0 || (content.count == 0 && uses.count == 0)) {
		[content addObject:@{ @"type": @"text", @"text": text ?: @"" }];
	}
	[content addObjectsFromArray:uses];
	return @{ @"id": identifier, @"type": @"message", @"role": @"assistant", @"model": model, @"content": content,
			  @"stop_reason": [self stopReasonForResult:result request:request callsTools:uses.count > 0],
			  @"stop_sequence": NSNull.null, @"usage": [self usageForResult:result] };
}

#pragma mark Errors

+ (NSString *)errorTypeForStatus:(NSInteger)status
{
	NSDictionary<NSNumber *, NSString *> *types = @{ @401: @"authentication_error", @403: @"permission_error",
													 @404: @"not_found_error", @413: @"request_too_large",
													 @421: @"permission_error", @429: @"rate_limit_error",
													 @503: @"overloaded_error", @529: @"overloaded_error" };
	return types[@(status)] ?: (status >= 500 ? @"api_error" : @"invalid_request_error");
}

+ (NSDictionary<NSString *, id> *)errorObjectForError:(NSError *)error status:(NSInteger)status
{
	NSMutableDictionary<NSString *, id> *detail = [[NFKInferenceWireCoding JSONObjectForError:error][@"error"] mutableCopy];
	[detail removeObjectForKey:@"code"];
	detail[@"type"] = [self errorTypeForStatus:status];
	return @{ @"type": @"error", @"error": detail };
}

@end

#pragma mark - Streaming

/*! The kind of content block open on a stream. */
typedef NS_ENUM(NSInteger, NFKServedAnthropicBlock) {
	NFKServedAnthropicBlockNone,
	NFKServedAnthropicBlockThinking,
	NFKServedAnthropicBlockText,
	NFKServedAnthropicBlockForcedTool,
	NFKServedAnthropicBlockToolUse,
};

@interface NFKServedAnthropicStream ()
@property (nonatomic, strong) NFKInferenceRequest *request;
@property (nonatomic, copy) NSString *identifier;
@property (nonatomic, copy) NSString *model;
@property (nonatomic, copy, nullable) NSString *forcedToolName;
@property (nonatomic, copy) NSString *forcedToolIdentifier;
@property (nonatomic, copy) void (^eventHandler)(NSString *name, NSDictionary<NSString *, id> *data);
@property (nonatomic, readwrite) BOOL started;
@property (nonatomic, assign) NFKServedAnthropicBlock openBlock;
@property (nonatomic, assign) NSInteger blockCount;
@property (nonatomic, assign) BOOL sentForcedTool;
@property (nonatomic, strong) NSMutableString *sentText;
@property (nonatomic, strong) NSMutableString *sentReasoning;
@end

@implementation NFKServedAnthropicStream

- (instancetype)initWithRequest:(NFKInferenceRequest *)request
					 identifier:(NSString *)identifier
						  model:(NSString *)model
				 forcedToolName:(nullable NSString *)forcedToolName
				   eventHandler:(void (^)(NSString *name, NSDictionary<NSString *, id> *data))eventHandler
{
	self = [super init];
	if (self != nil) {
		_request = request;
		_identifier = [identifier copy];
		_model = [model copy];
		_forcedToolName = [forcedToolName copy];
		_forcedToolIdentifier = NFKServedAnthropicIdentifier(@"toolu_");
		_eventHandler = [eventHandler copy];
		_sentText = [NSMutableString string];
		_sentReasoning = [NSMutableString string];
	}
	return self;
}

- (void)send:(NSString *)name data:(NSDictionary<NSString *, id> *)data
{
	self.eventHandler(name, data);
}

- (void)startWithUsage:(NSDictionary<NSString *, id> *)usage
{
	if (self.started) {
		return;
	}
	self.started = YES;
	NSDictionary *message = @{ @"id": self.identifier, @"type": @"message", @"role": @"assistant", @"model": self.model,
							   @"content": @[], @"stop_reason": NSNull.null, @"stop_sequence": NSNull.null, @"usage": usage };
	[self send:@"message_start" data:@{ @"type": @"message_start", @"message": message }];
}

- (void)closeBlock
{
	if (self.openBlock == NFKServedAnthropicBlockNone) {
		return;
	}
	[self send:@"content_block_stop" data:@{ @"type": @"content_block_stop", @"index": @(self.blockCount - 1) }];
	self.openBlock = NFKServedAnthropicBlockNone;
}

- (void)openBlock:(NFKServedAnthropicBlock)kind contentBlock:(NSDictionary *)contentBlock
{
	[self closeBlock];
	[self send:@"content_block_start" data:@{ @"type": @"content_block_start", @"index": @(self.blockCount), @"content_block": contentBlock }];
	self.blockCount += 1;
	self.openBlock = kind;
}

- (NSDictionary *)contentBlockForKind:(NFKServedAnthropicBlock)kind
{
	switch (kind) {
		case NFKServedAnthropicBlockThinking:
			return @{ @"type": @"thinking", @"thinking": @"", @"signature": @"" };
		case NFKServedAnthropicBlockForcedTool:
			return @{ @"type": @"tool_use", @"id": self.forcedToolIdentifier, @"name": self.forcedToolName ?: @"", @"input": @{} };
		default:
			return @{ @"type": @"text", @"text": @"" };
	}
}

- (NSDictionary *)deltaOfKind:(NFKServedAnthropicBlock)kind piece:(NSString *)piece
{
	switch (kind) {
		case NFKServedAnthropicBlockThinking:
			return @{ @"type": @"thinking_delta", @"thinking": piece };
		case NFKServedAnthropicBlockForcedTool:
			return @{ @"type": @"input_json_delta", @"partial_json": piece };
		default:
			return @{ @"type": @"text_delta", @"text": piece };
	}
}

- (void)sendPiece:(NSString *)piece kind:(NFKServedAnthropicBlock)kind
{
	[self startWithUsage:[NFKServedAnthropic usageForResult:nil]];
	if (self.openBlock != kind) {
		[self openBlock:kind contentBlock:[self contentBlockForKind:kind]];
	}
	if (kind == NFKServedAnthropicBlockForcedTool) {
		self.sentForcedTool = YES;
	}
	[self send:@"content_block_delta" data:@{ @"type": @"content_block_delta", @"index": @(self.blockCount - 1),
											  @"delta": [self deltaOfKind:kind piece:piece] }];
}

/*! Sends the reasoning and text past what was sent; the text is the forced tool's input when a tool
	is forced. */
- (void)advanceTo:(NFKInferenceResult *)result
{
	NSString *reasoning = NFKServedSuffix([NFKServedAnthropic reasoningForResult:result], self.sentReasoning);
	if (reasoning != nil) {
		[self.sentReasoning appendString:reasoning];
		[self sendPiece:reasoning kind:NFKServedAnthropicBlockThinking];
	}
	NSString *text = NFKServedSuffix(result.text, self.sentText);
	if (text != nil) {
		[self.sentText appendString:text];
		[self sendPiece:text kind:self.forcedToolName != nil ? NFKServedAnthropicBlockForcedTool : NFKServedAnthropicBlockText];
	}
}

- (void)appendPartial:(NFKInferenceResult *)partial
{
	[self advanceTo:partial];
}

- (void)finishWithResult:(NFKInferenceResult *)result
{
	NSDictionary *usage = [NFKServedAnthropic usageForResult:result];
	[self startWithUsage:usage];
	[self advanceTo:result];
	NSArray<NSDictionary *> *uses = self.forcedToolName != nil ? @[] : [NFKServedAnthropic toolUsesForResult:result forcedToolName:nil];
	if (self.forcedToolName == nil && self.sentText.length == 0) {
		NSString *text = [NFKServedOpenAI textForResult:result];
		if (text.length > 0 || (self.blockCount == 0 && uses.count == 0)) {
			[self.sentText appendString:text ?: @""];
			[self openBlock:NFKServedAnthropicBlockText contentBlock:[self contentBlockForKind:NFKServedAnthropicBlockText]];
			if (text.length > 0) {
				[self sendPiece:text kind:NFKServedAnthropicBlockText];
			}
		}
	}
	if (self.forcedToolName != nil && !self.sentForcedTool) {
		[self sendPiece:NFKServedAnthropicJSONText([NFKServedAnthropic forcedInputForResult:result]) kind:NFKServedAnthropicBlockForcedTool];
	}
	[self closeBlock];
	for (NSDictionary *use in uses) {
		[self openBlock:NFKServedAnthropicBlockToolUse contentBlock:@{ @"type": @"tool_use", @"id": use[@"id"], @"name": use[@"name"], @"input": @{} }];
		[self send:@"content_block_delta" data:@{ @"type": @"content_block_delta", @"index": @(self.blockCount - 1),
												  @"delta": @{ @"type": @"input_json_delta", @"partial_json": NFKServedAnthropicJSONText(use[@"input"]) } }];
		[self closeBlock];
	}
	BOOL callsTools = self.forcedToolName != nil || uses.count > 0;
	NSDictionary *delta = @{ @"stop_reason": [NFKServedAnthropic stopReasonForResult:result request:self.request callsTools:callsTools],
							 @"stop_sequence": NSNull.null };
	[self send:@"message_delta" data:@{ @"type": @"message_delta", @"delta": delta, @"usage": usage }];
	[self send:@"message_stop" data:@{ @"type": @"message_stop" }];
}

- (void)failWithError:(NSError *)error status:(NSInteger)status
{
	[self send:@"error" data:[NFKServedAnthropic errorObjectForError:error status:status]];
}

@end
