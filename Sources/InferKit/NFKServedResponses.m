//
//  NFKServedResponses.m
//  InferKit
//

#import "NFKServedResponses.h"
#import "NFKServedOpenAI.h"
#import "NFKInferenceServerPrivate.h"
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKInferenceTool.h>
#import <InferKit/NFKToolMessages.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKErrors.h>

static NSString *NFKServedResponsesIdentifier(NSString *prefix)
{
	NSString *digits = [NSUUID.UUID.UUIDString stringByReplacingOccurrencesOfString:@"-" withString:@""].lowercaseString;
	return [prefix stringByAppendingString:digits];
}

static NSString *NFKServedResponsesJSONText(id object)
{
	if (object == nil || ![NSJSONSerialization isValidJSONObject:object]) {
		return @"{}";
	}
	NSData *data = [NSJSONSerialization dataWithJSONObject:object options:0 error:NULL];
	return data != nil ? [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] : @"{}";
}

/*! The value under key when it is of class, otherwise nil. */
static id _Nullable NFKServedResponsesValue(NSDictionary *object, NSString *key, Class kind)
{
	id value = [object isKindOfClass:NSDictionary.class] ? object[key] : nil;
	return [value isKindOfClass:kind] ? value : nil;
}

/*! The parameters schema a custom tool's free-form input reaches the backend under. */
static NSDictionary *NFKServedCustomToolParameters(void)
{
	return @{ @"type": @"object",
			  @"properties": @{ @"input": @{ @"type": @"string", @"description": @"The tool's whole input as free-form text." } },
			  @"required": @[ @"input" ] };
}

/*! The assistant turn the items since the last other turn build: an assistant message's text and the
	calls that follow it. */
@interface NFKServedResponsesAssistantTurn : NSObject
@property (nonatomic, copy, nullable) NSString *text;
@property (nonatomic, strong) NSMutableArray<NSDictionary *> *calls;
@end

@implementation NFKServedResponsesAssistantTurn
- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_calls = [NSMutableArray array];
	}
	return self;
}
@end

@implementation NFKServedResponses

#pragma mark Routes

+ (void)serveResponses:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response server:(NFKInferenceServer *)server
{
	NSDictionary *body = [server JSONBodyOf:request response:response wire:NFKServedWireOpenAI];
	if (body == nil) {
		return;
	}
	NSError *error = nil;
	NSSet<NSString *> *customToolNames = nil;
	NFKServedModel *model = [server modelNamed:body[@"model"] error:&error];
	NFKInferenceRequest *inference = model != nil ? [self requestFromBody:body customToolNames:&customToolNames error:&error] : nil;
	if (inference == nil) {
		[server sendError:error response:response wire:NFKServedWireOpenAI];
		return;
	}
	NSString *identifier = [self newResponseIdentifier];
	NSInteger created = (NSInteger)NSDate.date.timeIntervalSince1970;
	NSString *modelName = NFKServedModelName(model);
	NSSet<NSString *> *customNames = customToolNames ?: [NSSet set];
	NFKServedRun *run = [[NFKServedRun alloc] init];
	run.request = inference;
	if (![body[@"stream"] boolValue]) {
		run.completionHandler = ^(NFKInferenceResult *result, NSError *runError) {
			if (result == nil) {
				[server sendError:runError response:response wire:NFKServedWireOpenAI];
				return;
			}
			[response sendStatus:200 JSONObject:[self responseForResult:result body:body identifier:identifier createdAt:created
																  model:modelName customToolNames:customNames] headers:nil];
		};
		[server enqueue:run on:model request:request response:response];
		return;
	}
	NFKServedResponsesStream *stream = [[NFKServedResponsesStream alloc] initWithBody:body identifier:identifier createdAt:created model:modelName
																	  customToolNames:customNames eventHandler:NFKServedEventWriter(response)];
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
				[server sendError:runError response:response wire:NFKServedWireOpenAI];
				return;
			}
			if (result != nil) {
				[stream finishWithResult:result];
			} else {
				[stream failWithError:runError ?: [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceBackendFailure
																  userInfo:@{ NSLocalizedDescriptionKey: @"the run failed" }]];
			}
			[response endStream];
		}
	};
	[server enqueue:run on:model request:request response:response];
}

+ (void)serveInputTokens:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response server:(NFKInferenceServer *)server
{
	NSDictionary *body = [server JSONBodyOf:request response:response wire:NFKServedWireOpenAI];
	if (body == nil) {
		return;
	}
	NSError *error = nil;
	NFKServedModel *model = [server modelNamed:body[@"model"] error:&error];
	NFKInferenceRequest *inference = model != nil ? [self requestFromBody:body customToolNames:NULL error:&error] : nil;
	if (inference == nil) {
		[server sendError:error response:response wire:NFKServedWireOpenAI];
		return;
	}
	NFKServedRun *run = [NFKServedRun tokenCountRunForRequest:inference modelName:NFKServedModelName(model)];
	run.completionHandler = ^(NFKInferenceResult *result, NSError *runError) {
		NSNumber *count = result != nil ? [result outputForKey:NFKOutputUsage][NFKUsageInputTokens] : nil;
		if (count == nil) {
			[server sendError:runError response:response wire:NFKServedWireOpenAI];
			return;
		}
		[response sendStatus:200 JSONObject:@{ @"object": @"response.input_tokens", @"input_tokens": count } headers:nil];
	};
	[server enqueue:run on:model request:request response:response];
}

#pragma mark Request

+ (nullable NFKInferenceRequest *)requestFromBody:(NSDictionary *)body
								  customToolNames:(NSSet<NSString *> * _Nullable * _Nullable)outCustomToolNames
											error:(NSError * _Nullable *)outError
{
	if (NFKServedResponsesValue(body, @"previous_response_id", NSString.class) != nil || (body[@"conversation"] != nil && body[@"conversation"] != NSNull.null)) {
		[NFKServedOpenAI setError:outError code:kNFKError_InferenceUnsupported
						   reason:@"this server keeps no responses or conversations; send the whole conversation as input"];
		return nil;
	}
	NSMutableArray<NSDictionary *> *messages = [NSMutableArray array];
	NSString *instructions = NFKServedResponsesValue(body, @"instructions", NSString.class);
	if (instructions.length > 0) {
		[messages addObject:@{ @"role": @"system", @"content": instructions }];
	}
	NSMutableArray *images = [NSMutableArray array];
	NSMutableArray *documents = [NSMutableArray array];
	id input = body[@"input"];
	if ([input isKindOfClass:NSString.class] && [input length] > 0) {
		[messages addObject:@{ @"role": @"user", @"content": input }];
	} else if ([input isKindOfClass:NSArray.class] && ![self readItems:input into:messages images:images documents:documents error:outError]) {
		return nil;
	}
	if ([[messages valueForKey:@"role"] indexOfObjectPassingTest:^BOOL(id role, NSUInteger index, BOOL *stop) {
		return ![role isEqual:@"system"];
	}] == NSNotFound) {
		[NFKServedOpenAI setError:outError code:kNFKError_InferenceMissingInput reason:@"the request carries no input"];
		return nil;
	}
	NSArray<NSDictionary *> *conversation = [self conversationWithLeadingSystemTurnsJoined:messages];

	NSMutableSet<NSString *> *customNames = [NSMutableSet set];
	NSDictionary *parameters = [self parametersFromBody:body customToolNames:customNames];
	if (outCustomToolNames != NULL) {
		*outCustomToolNames = customNames;
	}
	NSMutableDictionary<NSString *, id> *inputs = [NSMutableDictionary dictionaryWithObject:conversation forKey:NFKInputMessages];
	NSDictionary *only = conversation.count == 1 ? conversation.firstObject : nil;
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

// Many chat templates take one system turn, first; Codex sends its instructions and its developer
// turns separately, so the run of them before the first other turn becomes one.
+ (NSArray<NSDictionary *> *)conversationWithLeadingSystemTurnsJoined:(NSArray<NSDictionary *> *)messages
{
	NSUInteger leading = 0;
	NSMutableArray<NSString *> *texts = [NSMutableArray array];
	while (leading < messages.count && [messages[leading][@"role"] isEqual:@"system"]) {
		[texts addObject:messages[leading][@"content"]];
		leading += 1;
	}
	if (leading < 2) {
		return messages;
	}
	NSMutableArray<NSDictionary *> *joined = [NSMutableArray arrayWithObject:@{ @"role": @"system", @"content": [texts componentsJoinedByString:@"\n\n"] }];
	[joined addObjectsFromArray:[messages subarrayWithRange:NSMakeRange(leading, messages.count - leading)]];
	return joined;
}

+ (BOOL)readItems:(NSArray *)items
			 into:(NSMutableArray<NSDictionary *> *)messages
		   images:(NSMutableArray *)images
		documents:(NSMutableArray *)documents
			error:(NSError * _Nullable *)outError
{
	NSMutableDictionary<NSString *, NSString *> *toolNames = [NSMutableDictionary dictionary];
	NFKServedResponsesAssistantTurn *turn = nil;
	for (NSDictionary *item in items) {
		if (![item isKindOfClass:NSDictionary.class]) {
			return [NFKServedOpenAI setError:outError code:kNFKError_InferenceMissingInput reason:@"an input item is not an object"];
		}
		NSString *type = NFKServedResponsesValue(item, @"type", NSString.class) ?: @"message";
		if ([type isEqualToString:@"reasoning"]) {
			continue;
		}
		if ([type isEqualToString:@"function_call"] || [type isEqualToString:@"custom_tool_call"]) {
			NSDictionary *call = [self callForItem:item custom:[type isEqualToString:@"custom_tool_call"]];
			toolNames[call[@"id"]] = call[@"name"];
			turn = turn ?: [[NFKServedResponsesAssistantTurn alloc] init];
			[turn.calls addObject:call];
			continue;
		}
		[self flushTurn:turn into:messages];
		turn = nil;
		if ([type isEqualToString:@"function_call_output"] || [type isEqualToString:@"custom_tool_call_output"]) {
			NSDictionary *answer = [self toolTurnForItem:item toolNames:toolNames error:outError];
			if (answer == nil) {
				return NO;
			}
			[messages addObject:answer];
			continue;
		}
		if (![type isEqualToString:@"message"]) {
			return [NFKServedOpenAI setError:outError code:kNFKError_InferenceUnsupported
									  reason:[NSString stringWithFormat:@"the input item type \"%@\" is not served; this server keeps no items", type]];
		}
		NSString *role = NFKServedResponsesValue(item, @"role", NSString.class) ?: @"user";
		NSString *text = [self textOfContent:item[@"content"] images:images documents:documents error:outError];
		if (text == nil) {
			return NO;
		}
		if ([role isEqualToString:@"assistant"]) {
			turn = [[NFKServedResponsesAssistantTurn alloc] init];
			turn.text = text;
			continue;
		}
		[messages addObject:@{ @"role": [role isEqualToString:@"developer"] ? @"system" : role, @"content": text }];
	}
	[self flushTurn:turn into:messages];
	return YES;
}

+ (void)flushTurn:(nullable NFKServedResponsesAssistantTurn *)turn into:(NSMutableArray<NSDictionary *> *)messages
{
	if (turn == nil) {
		return;
	}
	[messages addObject:[NFKToolMessages assistantMessageWithText:turn.text toolCalls:turn.calls record:nil format:nil source:nil]];
}

/*! A call item in the NFKOutputToolCalls shape. A custom tool's free-form input becomes the input
	argument. */
+ (NSDictionary *)callForItem:(NSDictionary *)item custom:(BOOL)custom
{
	NSString *identifier = NFKServedResponsesValue(item, @"call_id", NSString.class) ?: NFKServedResponsesValue(item, @"id", NSString.class) ?: @"";
	NSString *name = NFKServedResponsesValue(item, @"name", NSString.class) ?: @"";
	NSDictionary *arguments = nil;
	NSString *argumentsJSON = nil;
	if (custom) {
		arguments = @{ @"input": NFKServedResponsesValue(item, @"input", NSString.class) ?: @"" };
		argumentsJSON = NFKServedResponsesJSONText(arguments);
	} else {
		argumentsJSON = NFKServedResponsesValue(item, @"arguments", NSString.class) ?: @"{}";
		id parsed = [NSJSONSerialization JSONObjectWithData:[argumentsJSON dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
		arguments = [parsed isKindOfClass:NSDictionary.class] ? parsed : @{};
	}
	return @{ @"id": identifier, @"name": name, @"arguments": arguments, @"argumentsJSON": argumentsJSON };
}

+ (nullable NSDictionary *)toolTurnForItem:(NSDictionary *)item
								 toolNames:(NSDictionary<NSString *, NSString *> *)toolNames
									 error:(NSError * _Nullable *)outError
{
	NSString *identifier = NFKServedResponsesValue(item, @"call_id", NSString.class);
	NSString *name = identifier != nil ? toolNames[identifier] : nil;
	if (name == nil) {
		[NFKServedOpenAI setError:outError code:kNFKError_InferenceMissingInput
						   reason:@"a call output names a call_id that no earlier call item carries"];
		return nil;
	}
	id output = item[@"output"];
	NSMutableArray<NSString *> *texts = [NSMutableArray array];
	NSMutableArray<NSData *> *pngs = [NSMutableArray array];
	if ([output isKindOfClass:NSString.class]) {
		[texts addObject:output];
	}
	for (NSDictionary *part in [output isKindOfClass:NSArray.class] ? output : @[]) {
		NSString *type = NFKServedResponsesValue(part, @"type", NSString.class);
		if (([type isEqualToString:@"input_text"] || [type isEqualToString:@"output_text"]) && NFKServedResponsesValue(part, @"text", NSString.class) != nil) {
			[texts addObject:part[@"text"]];
			continue;
		}
		id image = [type isEqualToString:@"input_image"] ? [self imageForPart:part error:outError] : nil;
		NSData *png = image != nil ? [NFKImageCoding PNGDataForImage:image] : nil;
		if (png == nil) {
			if (image == nil && [type isEqualToString:@"input_image"]) {
				return nil;
			}
			[NFKServedOpenAI setError:outError code:kNFKError_InferenceUnsupported
							   reason:[NSString stringWithFormat:@"a call output part of type \"%@\" is not served", type ?: @"(none)"]];
			return nil;
		}
		[pngs addObject:png];
	}
	NFKToolResult *result = [[NFKToolResult alloc] initWithText:[texts componentsJoinedByString:@"\n"] outputs:nil imagePNGs:pngs isError:NO];
	return [NFKToolMessages messageForToolCallIdentifier:identifier name:name result:result];
}

/*! A message's text, its text parts joined; its images and files join the request's media. */
+ (nullable NSString *)textOfContent:(id)content
							  images:(NSMutableArray *)images
						   documents:(NSMutableArray *)documents
							   error:(NSError * _Nullable *)outError
{
	if ([content isKindOfClass:NSString.class]) {
		return content;
	}
	NSMutableArray<NSString *> *texts = [NSMutableArray array];
	for (NSDictionary *part in [content isKindOfClass:NSArray.class] ? content : @[]) {
		NSString *type = NFKServedResponsesValue(part, @"type", NSString.class) ?: @"";
		if ([@[ @"input_text", @"output_text", @"text" ] containsObject:type]) {
			[texts addObject:NFKServedResponsesValue(part, @"text", NSString.class) ?: @""];
		} else if ([type isEqualToString:@"refusal"]) {
			[texts addObject:NFKServedResponsesValue(part, @"refusal", NSString.class) ?: @""];
		} else if ([type isEqualToString:@"input_image"]) {
			id image = [self imageForPart:part error:outError];
			if (image == nil) {
				return nil;
			}
			[images addObject:image];
		} else if ([type isEqualToString:@"input_file"]) {
			NSData *bytes = [self bytesOfFilePart:part];
			if (bytes == nil) {
				[NFKServedOpenAI setError:outError code:kNFKError_InferenceUnsupported
								   reason:@"a file part must carry file_data inline; this server keeps no files and fetches no URL"];
				return nil;
			}
			[documents addObject:bytes];
		} else {
			[NFKServedOpenAI setError:outError code:kNFKError_InferenceUnsupported
							   reason:[NSString stringWithFormat:@"the content part type \"%@\" is not served", type]];
			return nil;
		}
	}
	return [texts componentsJoinedByString:@"\n"];
}

+ (nullable id)imageForPart:(NSDictionary *)part error:(NSError * _Nullable *)outError
{
	id reference = part[@"image_url"];
	id url = [reference isKindOfClass:NSDictionary.class] ? reference[@"url"] : reference;
	NSData *bytes = NFKServedDataURLBytes(url, NULL);
	CVPixelBufferRef buffer = bytes != nil ? [NFKImageCoding pixelBufferWithImageData:bytes] : NULL;
	if (buffer == NULL) {
		[NFKServedOpenAI setError:outError code:kNFKError_InferenceUnsupported
						   reason:@"an image part must be an inline data URL of an image this server can decode; this server keeps no files"];
		return nil;
	}
	return (__bridge_transfer id)buffer;
}

+ (nullable NSData *)bytesOfFilePart:(NSDictionary *)part
{
	NSString *data = NFKServedResponsesValue(part, @"file_data", NSString.class);
	if ([data hasPrefix:@"data:"]) {
		return NFKServedDataURLBytes(data, NULL);
	}
	return data != nil ? [[NSData alloc] initWithBase64EncodedString:data options:NSDataBase64DecodingIgnoreUnknownCharacters] : nil;
}

#pragma mark Parameters

+ (NSDictionary<NSString *, id> *)parametersFromBody:(NSDictionary *)body customToolNames:(NSMutableSet<NSString *> *)customToolNames
{
	NSMutableDictionary<NSString *, id> *parameters = [NSMutableDictionary dictionary];
	NSSet<NSString *> *consumed = [NSSet setWithArray:@[ @"model", @"input", @"instructions", @"stream", @"stream_options", @"tools",
														 @"tool_choice", @"temperature", @"top_p", @"max_output_tokens", @"reasoning",
														 @"text", @"prompt_cache_key", @"previous_response_id", @"conversation", @"store",
														 @"include", @"background", @"metadata", @"user", @"safety_identifier",
														 @"service_tier", @"truncation" ]];
	for (NSString *key in body) {
		if (![consumed containsObject:key]) {
			parameters[key] = body[key];
		}
	}
	NSDictionary<NSString *, NSString *> *renamed = @{ @"temperature": NFKParameterTemperature, @"top_p": NFKParameterTopP,
													   @"max_output_tokens": NFKParameterMaxTokens };
	for (NSString *key in renamed) {
		if ([body[key] isKindOfClass:NSNumber.class]) {
			parameters[renamed[key]] = body[key];
		}
	}
	parameters[NFKParameterConversationKey] = NFKServedResponsesValue(body, @"prompt_cache_key", NSString.class);
	NSString *effort = NFKServedResponsesValue(NFKServedResponsesValue(body, @"reasoning", NSDictionary.class), @"effort", NSString.class);
	if (effort != nil) {
		NSDictionary<NSString *, NSString *> *levels = @{ @"low": NFKReasoningEffortLight, @"medium": NFKReasoningEffortModerate,
														  @"high": NFKReasoningEffortDeep };
		parameters[NFKParameterReasoningEffort] = levels[effort] ?: effort;
	}
	NSDictionary *format = NFKServedResponsesValue(NFKServedResponsesValue(body, @"text", NSDictionary.class), @"format", NSDictionary.class);
	NSDictionary *schema = NFKServedResponsesValue(format, @"schema", NSDictionary.class);
	if ([format[@"type"] isEqual:@"json_schema"] && schema != nil) {
		parameters[NFKParameterJSONSchema] = schema;
	} else if ([format[@"type"] isEqual:@"json_object"]) {
		parameters[@"response_format"] = @{ @"type": @"json_object" };
	}
	NSArray *tools = [self contractToolsOf:body[@"tools"] customToolNames:customToolNames];
	if (tools.count > 0) {
		parameters[NFKParameterTools] = tools;
	}
	id choice = body[@"tool_choice"];
	NSString *named = NFKServedResponsesValue(choice, @"name", NSString.class);
	if ([choice isKindOfClass:NSString.class]) {
		parameters[@"tool_choice"] = choice;
	} else if (named != nil) {
		parameters[@"tool_choice"] = @{ @"type": @"function", @"function": @{ @"name": named } };
	}
	return parameters;
}

// A built-in tool (web search, file search, a local shell, image generation, MCP) runs at OpenAI;
// this server runs none, so only function and custom tools are offered.
+ (NSArray<NSDictionary *> *)contractToolsOf:(id)wireTools customToolNames:(NSMutableSet<NSString *> *)customToolNames
{
	NSMutableArray<NSDictionary *> *tools = [NSMutableArray array];
	for (NSDictionary *tool in [wireTools isKindOfClass:NSArray.class] ? wireTools : @[]) {
		NSString *name = NFKServedResponsesValue(tool, @"name", NSString.class);
		NSString *type = NFKServedResponsesValue(tool, @"type", NSString.class);
		if (name == nil || !([type isEqualToString:@"function"] || [type isEqualToString:@"custom"])) {
			continue;
		}
		NSMutableDictionary *contract = [NSMutableDictionary dictionaryWithObject:name forKey:@"name"];
		contract[@"description"] = NFKServedResponsesValue(tool, @"description", NSString.class);
		if ([type isEqualToString:@"custom"]) {
			contract[@"parameters"] = NFKServedCustomToolParameters();
			[customToolNames addObject:name];
		} else {
			contract[@"parameters"] = NFKServedResponsesValue(tool, @"parameters", NSDictionary.class) ?: @{ @"type": @"object", @"properties": @{} };
		}
		[tools addObject:contract];
	}
	return tools;
}

#pragma mark Reply

+ (NSString *)newResponseIdentifier
{
	return NFKServedResponsesIdentifier(@"resp_");
}

+ (nullable NSString *)reasoningForResult:(nullable NFKInferenceResult *)result
{
	NSString *reasoning = [result outputForKey:NFKOutputReasoning];
	return [reasoning isKindOfClass:NSString.class] ? reasoning : nil;
}

+ (NSDictionary *)reasoningItemWithIdentifier:(NSString *)identifier text:(nullable NSString *)text
{
	NSArray *summary = text != nil ? @[ @{ @"type": @"summary_text", @"text": text } ] : @[];
	return @{ @"type": @"reasoning", @"id": identifier, @"summary": summary };
}

+ (NSDictionary *)textPartWithText:(NSString *)text annotations:(NSArray *)annotations
{
	return @{ @"type": @"output_text", @"text": text, @"annotations": annotations };
}

+ (NSDictionary *)messageItemWithIdentifier:(NSString *)identifier parts:(NSArray *)parts completed:(BOOL)completed
{
	return @{ @"type": @"message", @"id": identifier, @"status": completed ? @"completed" : @"in_progress",
			  @"role": @"assistant", @"content": parts };
}

+ (NSArray<NSDictionary *> *)annotationsForResult:(nullable NFKInferenceResult *)result
{
	NSArray *citations = [result outputForKey:NFKOutputCitations];
	NSMutableArray<NSDictionary *> *annotations = [NSMutableArray array];
	for (NSDictionary *citation in [citations isKindOfClass:NSArray.class] ? citations : @[]) {
		NSString *url = NFKServedResponsesValue(citation, @"url", NSString.class);
		if (url == nil) {
			continue;
		}
		NSMutableDictionary *annotation = [NSMutableDictionary dictionaryWithDictionary:@{ @"type": @"url_citation", @"url": url }];
		annotation[@"title"] = citation[@"title"];
		annotation[@"start_index"] = citation[@"start"];
		annotation[@"end_index"] = citation[@"end"];
		[annotations addObject:annotation];
	}
	return annotations;
}

/*! The call items a result's NFKOutputToolCalls become, completed; a call to a custom tool is a
	custom_tool_call whose input is its input argument. */
+ (NSArray<NSDictionary *> *)callItemsForResult:(NFKInferenceResult *)result customToolNames:(NSSet<NSString *> *)customToolNames
{
	NSMutableArray<NSDictionary *> *items = [NSMutableArray array];
	for (NSDictionary *call in result.toolCalls ?: @[]) {
		NSString *name = NFKServedResponsesValue(call, @"name", NSString.class);
		if (name == nil) {
			continue;
		}
		NSString *identifier = [NFKServedResponsesValue(call, @"id", NSString.class) length] > 0 ? call[@"id"] : NFKServedResponsesIdentifier(@"call_");
		NSDictionary *arguments = NFKServedResponsesValue(call, @"arguments", NSDictionary.class);
		NSString *argumentsJSON = NFKServedResponsesValue(call, @"argumentsJSON", NSString.class) ?: NFKServedResponsesJSONText(arguments ?: @{});
		if ([customToolNames containsObject:name]) {
			NSString *input = NFKServedResponsesValue(arguments, @"input", NSString.class) ?: argumentsJSON;
			[items addObject:@{ @"type": @"custom_tool_call", @"id": NFKServedResponsesIdentifier(@"ctc_"), @"call_id": identifier,
								@"name": name, @"input": input, @"status": @"completed" }];
		} else {
			[items addObject:@{ @"type": @"function_call", @"id": NFKServedResponsesIdentifier(@"fc_"), @"call_id": identifier,
								@"name": name, @"arguments": argumentsJSON, @"status": @"completed" }];
		}
	}
	return items;
}

/*! The usage object, or NSNull when the backend reports none. */
+ (id)usageForResult:(nullable NFKInferenceResult *)result
{
	NSDictionary *usage = [result outputForKey:NFKOutputUsage];
	if (![usage isKindOfClass:NSDictionary.class] || usage.count == 0) {
		return NSNull.null;
	}
	NSInteger input = [usage[NFKUsageInputTokens] integerValue];
	NSInteger output = [usage[NFKUsageOutputTokens] integerValue];
	NSMutableDictionary *wire = [@{ @"input_tokens": @(input), @"output_tokens": @(output), @"total_tokens": @(input + output) } mutableCopy];
	if ([usage[NFKUsageCachedTokens] isKindOfClass:NSNumber.class]) {
		wire[@"input_tokens_details"] = @{ @"cached_tokens": usage[NFKUsageCachedTokens] };
	}
	if ([usage[NFKUsageReasoningTokens] isKindOfClass:NSNumber.class]) {
		wire[@"output_tokens_details"] = @{ @"reasoning_tokens": usage[NFKUsageReasoningTokens] };
	}
	return wire;
}

+ (BOOL)resultReachedLimit:(NFKInferenceResult *)result body:(NSDictionary *)body
{
	NSNumber *limit = NFKServedResponsesValue(body, @"max_output_tokens", NSNumber.class);
	NSDictionary *usage = [result outputForKey:NFKOutputUsage];
	NSNumber *produced = [usage isKindOfClass:NSDictionary.class] ? usage[NFKUsageOutputTokens] : nil;
	return limit.integerValue > 0 && [produced isKindOfClass:NSNumber.class] && produced.integerValue >= limit.integerValue;
}

/*! The response object in a given state, echoing what the request set. */
+ (NSDictionary<NSString *, id> *)responseWithBody:(NSDictionary *)body
										identifier:(NSString *)identifier
										 createdAt:(NSInteger)createdAt
											 model:(NSString *)model
											status:(NSString *)status
											output:(NSArray *)output
											 usage:(id)usage
											 error:(nullable NSDictionary *)error
{
	NSDictionary *reasoning = NFKServedResponsesValue(body, @"reasoning", NSDictionary.class);
	id (^echo)(NSString *, id) = ^id(NSString *key, id fallback) {
		id value = body[key];
		return value != nil ? value : fallback;
	};
	return @{ @"id": identifier, @"object": @"response", @"created_at": @(createdAt), @"status": status,
			  @"background": @NO, @"error": error ?: NSNull.null,
			  @"incomplete_details": [status isEqualToString:@"incomplete"] ? @{ @"reason": @"max_output_tokens" } : NSNull.null,
			  @"instructions": echo(@"instructions", NSNull.null), @"max_output_tokens": echo(@"max_output_tokens", NSNull.null),
			  @"model": model, @"output": output, @"parallel_tool_calls": echo(@"parallel_tool_calls", @YES),
			  @"previous_response_id": NSNull.null,
			  @"reasoning": @{ @"effort": reasoning[@"effort"] ?: NSNull.null, @"summary": reasoning[@"summary"] ?: NSNull.null },
			  @"store": @NO, @"temperature": echo(@"temperature", NSNull.null),
			  @"text": echo(@"text", @{ @"format": @{ @"type": @"text" } }), @"tool_choice": echo(@"tool_choice", @"auto"),
			  @"tools": echo(@"tools", @[]), @"top_p": echo(@"top_p", NSNull.null), @"truncation": @"disabled",
			  @"usage": usage, @"metadata": echo(@"metadata", @{}) };
}

+ (NSDictionary<NSString *, id> *)responseForResult:(NFKInferenceResult *)result
											   body:(NSDictionary *)body
										 identifier:(NSString *)identifier
										  createdAt:(NSInteger)createdAt
											  model:(NSString *)model
									customToolNames:(NSSet<NSString *> *)customToolNames
{
	NSMutableArray<NSDictionary *> *output = [NSMutableArray array];
	NSString *reasoning = [self reasoningForResult:result];
	if (reasoning.length > 0) {
		[output addObject:[self reasoningItemWithIdentifier:NFKServedResponsesIdentifier(@"rs_") text:reasoning]];
	}
	NSArray<NSDictionary *> *calls = [self callItemsForResult:result customToolNames:customToolNames];
	NSString *text = [NFKServedOpenAI textForResult:result];
	if (text.length > 0 || (output.count == 0 && calls.count == 0)) {
		NSDictionary *part = [self textPartWithText:text ?: @"" annotations:[self annotationsForResult:result]];
		[output addObject:[self messageItemWithIdentifier:NFKServedResponsesIdentifier(@"msg_") parts:@[ part ] completed:YES]];
	}
	[output addObjectsFromArray:calls];
	NSString *status = [self resultReachedLimit:result body:body] ? @"incomplete" : @"completed";
	return [self responseWithBody:body identifier:identifier createdAt:createdAt model:model status:status output:output
							usage:[self usageForResult:result] error:nil];
}

/*! The error a failed response carries: an OpenAI code, the message, and the InferKit domain and
	code. */
+ (NSDictionary *)failureForError:(NSError *)error
{
	NSString *code = @"server_error";
	if ([error.domain isEqualToString:NFKInferenceErrorDomain] && error.code == kNFKError_InferenceRateLimited) {
		code = @"rate_limit_exceeded";
	} else if ([error.domain isEqualToString:NFKInferenceErrorDomain]
			   && (error.code == kNFKError_InferenceMissingInput || error.code == kNFKError_InferenceUnsupported || error.code == kNFKError_InferenceRefused)) {
		code = @"invalid_prompt";
	}
	return @{ @"code": code, @"message": error.localizedDescription ?: @"the run failed",
			  @"inferkit_domain": error.domain, @"inferkit_code": @(error.code) };
}

@end

#pragma mark - Streaming

/*! The kind of output item open on a stream. */
typedef NS_ENUM(NSInteger, NFKServedResponsesItem) {
	NFKServedResponsesItemNone,
	NFKServedResponsesItemReasoning,
	NFKServedResponsesItemMessage,
};

@interface NFKServedResponsesStream ()
@property (nonatomic, copy) NSDictionary *body;
@property (nonatomic, copy) NSString *identifier;
@property (nonatomic, assign) NSInteger createdAt;
@property (nonatomic, copy) NSString *model;
@property (nonatomic, copy) NSSet<NSString *> *customToolNames;
@property (nonatomic, copy) void (^eventHandler)(NSString *name, NSDictionary<NSString *, id> *data);
@property (nonatomic, readwrite) BOOL started;
@property (nonatomic, assign) NSInteger sequenceNumber;
@property (nonatomic, strong) NSMutableArray<NSDictionary *> *items;
@property (nonatomic, assign) NFKServedResponsesItem openItem;
@property (nonatomic, copy) NSString *openIdentifier;
@property (nonatomic, strong) NSMutableString *openText;
@property (nonatomic, strong) NSMutableString *sentText;
@property (nonatomic, strong) NSMutableString *sentReasoning;
/*! The finished result, whose citations the message item closes with. */
@property (nonatomic, strong, nullable) NFKInferenceResult *finalResult;
@end

@implementation NFKServedResponsesStream

- (instancetype)initWithBody:(NSDictionary *)body
				  identifier:(NSString *)identifier
				   createdAt:(NSInteger)createdAt
					   model:(NSString *)model
			 customToolNames:(NSSet<NSString *> *)customToolNames
				eventHandler:(void (^)(NSString *name, NSDictionary<NSString *, id> *data))eventHandler
{
	self = [super init];
	if (self != nil) {
		_body = [body copy];
		_identifier = [identifier copy];
		_createdAt = createdAt;
		_model = [model copy];
		_customToolNames = [customToolNames copy];
		_eventHandler = [eventHandler copy];
		_items = [NSMutableArray array];
		_openIdentifier = @"";
		_openText = [NSMutableString string];
		_sentText = [NSMutableString string];
		_sentReasoning = [NSMutableString string];
	}
	return self;
}

- (void)send:(NSString *)type fields:(NSDictionary<NSString *, id> *)fields
{
	NSMutableDictionary<NSString *, id> *event = [fields mutableCopy];
	event[@"type"] = type;
	event[@"sequence_number"] = @(self.sequenceNumber);
	self.sequenceNumber += 1;
	self.eventHandler(type, event);
}

- (NSDictionary *)responseWithStatus:(NSString *)status usage:(id)usage error:(nullable NSDictionary *)error
{
	return [NFKServedResponses responseWithBody:self.body identifier:self.identifier createdAt:self.createdAt model:self.model
										 status:status output:[self.items copy] usage:usage error:error];
}

- (void)start
{
	if (self.started) {
		return;
	}
	self.started = YES;
	NSDictionary *response = [self responseWithStatus:@"in_progress" usage:NSNull.null error:nil];
	[self send:@"response.created" fields:@{ @"response": response }];
	[self send:@"response.in_progress" fields:@{ @"response": response }];
}

- (NSNumber *)outputIndex
{
	return @(self.items.count);
}

- (void)openItemOfKind:(NFKServedResponsesItem)kind
{
	[self closeItem];
	self.openItem = kind;
	[self.openText setString:@""];
	if (kind == NFKServedResponsesItemReasoning) {
		self.openIdentifier = NFKServedResponsesIdentifier(@"rs_");
		[self send:@"response.output_item.added" fields:@{ @"output_index": self.outputIndex,
														   @"item": [NFKServedResponses reasoningItemWithIdentifier:self.openIdentifier text:nil] }];
		[self send:@"response.reasoning_summary_part.added" fields:@{ @"item_id": self.openIdentifier, @"output_index": self.outputIndex,
																	   @"summary_index": @0, @"part": @{ @"type": @"summary_text", @"text": @"" } }];
		return;
	}
	self.openIdentifier = NFKServedResponsesIdentifier(@"msg_");
	[self send:@"response.output_item.added" fields:@{ @"output_index": self.outputIndex,
													   @"item": [NFKServedResponses messageItemWithIdentifier:self.openIdentifier parts:@[] completed:NO] }];
	[self send:@"response.content_part.added" fields:@{ @"item_id": self.openIdentifier, @"output_index": self.outputIndex, @"content_index": @0,
														@"part": [NFKServedResponses textPartWithText:@"" annotations:@[]] }];
}

- (void)closeItem
{
	NSString *text = [self.openText copy];
	if (self.openItem == NFKServedResponsesItemReasoning) {
		NSDictionary *part = @{ @"type": @"summary_text", @"text": text };
		[self send:@"response.reasoning_summary_text.done" fields:@{ @"item_id": self.openIdentifier, @"output_index": self.outputIndex,
																	 @"summary_index": @0, @"text": text }];
		[self send:@"response.reasoning_summary_part.done" fields:@{ @"item_id": self.openIdentifier, @"output_index": self.outputIndex,
																	 @"summary_index": @0, @"part": part }];
		[self finishItem:[NFKServedResponses reasoningItemWithIdentifier:self.openIdentifier text:text]];
	} else if (self.openItem == NFKServedResponsesItemMessage) {
		NSDictionary *part = [NFKServedResponses textPartWithText:text annotations:[NFKServedResponses annotationsForResult:self.finalResult]];
		[self send:@"response.output_text.done" fields:@{ @"item_id": self.openIdentifier, @"output_index": self.outputIndex,
														  @"content_index": @0, @"text": text, @"logprobs": @[] }];
		[self send:@"response.content_part.done" fields:@{ @"item_id": self.openIdentifier, @"output_index": self.outputIndex,
														   @"content_index": @0, @"part": part }];
		[self finishItem:[NFKServedResponses messageItemWithIdentifier:self.openIdentifier parts:@[ part ] completed:YES]];
	}
	self.openItem = NFKServedResponsesItemNone;
}

- (void)finishItem:(NSDictionary *)item
{
	[self send:@"response.output_item.done" fields:@{ @"output_index": self.outputIndex, @"item": item }];
	[self.items addObject:item];
}

- (void)sendPiece:(NSString *)piece kind:(NFKServedResponsesItem)kind
{
	[self start];
	if (self.openItem != kind) {
		[self openItemOfKind:kind];
	}
	[self.openText appendString:piece];
	if (kind == NFKServedResponsesItemReasoning) {
		[self send:@"response.reasoning_summary_text.delta" fields:@{ @"item_id": self.openIdentifier, @"output_index": self.outputIndex,
																	   @"summary_index": @0, @"delta": piece }];
	} else {
		[self send:@"response.output_text.delta" fields:@{ @"item_id": self.openIdentifier, @"output_index": self.outputIndex,
														   @"content_index": @0, @"delta": piece, @"logprobs": @[] }];
	}
}

- (void)advanceTo:(NFKInferenceResult *)result
{
	NSString *reasoning = NFKServedSuffix([NFKServedResponses reasoningForResult:result], self.sentReasoning);
	if (reasoning != nil) {
		[self.sentReasoning appendString:reasoning];
		[self sendPiece:reasoning kind:NFKServedResponsesItemReasoning];
	}
	NSString *text = NFKServedSuffix(result.text, self.sentText);
	if (text != nil) {
		[self.sentText appendString:text];
		[self sendPiece:text kind:NFKServedResponsesItemMessage];
	}
}

- (void)appendPartial:(NFKInferenceResult *)partial
{
	[self advanceTo:partial];
}

- (void)sendCallItem:(NSDictionary *)item
{
	BOOL custom = [item[@"type"] isEqual:@"custom_tool_call"];
	NSString *field = custom ? @"input" : @"arguments";
	NSMutableDictionary *announced = [item mutableCopy];
	announced[field] = @"";
	announced[@"status"] = @"in_progress";
	[self send:@"response.output_item.added" fields:@{ @"output_index": self.outputIndex, @"item": announced }];
	NSString *prefix = custom ? @"response.custom_tool_call_input" : @"response.function_call_arguments";
	[self send:[prefix stringByAppendingString:@".delta"] fields:@{ @"item_id": item[@"id"], @"output_index": self.outputIndex, @"delta": item[field] }];
	[self send:[prefix stringByAppendingString:@".done"] fields:@{ @"item_id": item[@"id"], @"output_index": self.outputIndex, field: item[field] }];
	[self finishItem:item];
}

- (void)finishWithResult:(NFKInferenceResult *)result
{
	self.finalResult = result;
	[self start];
	[self advanceTo:result];
	NSArray<NSDictionary *> *calls = [NFKServedResponses callItemsForResult:result customToolNames:self.customToolNames];
	NSString *text = [NFKServedOpenAI textForResult:result];
	if (self.sentText.length == 0 && (text.length > 0 || (self.items.count == 0 && self.openItem == NFKServedResponsesItemNone && calls.count == 0))) {
		[self.sentText appendString:text ?: @""];
		[self openItemOfKind:NFKServedResponsesItemMessage];
		if (text.length > 0) {
			[self sendPiece:text kind:NFKServedResponsesItemMessage];
		}
	}
	[self closeItem];
	for (NSDictionary *call in calls) {
		[self sendCallItem:call];
	}
	BOOL incomplete = [NFKServedResponses resultReachedLimit:result body:self.body];
	NSDictionary *response = [self responseWithStatus:incomplete ? @"incomplete" : @"completed"
												usage:[NFKServedResponses usageForResult:result] error:nil];
	[self send:incomplete ? @"response.incomplete" : @"response.completed" fields:@{ @"response": response }];
}

- (void)failWithError:(NSError *)error
{
	[self closeItem];
	NSDictionary *response = [self responseWithStatus:@"failed" usage:NSNull.null error:[NFKServedResponses failureForError:error]];
	[self send:@"response.failed" fields:@{ @"response": response }];
}

@end
