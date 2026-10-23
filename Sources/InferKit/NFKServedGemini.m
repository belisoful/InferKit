//
//  NFKServedGemini.m
//  InferKit
//

#import "NFKServedGemini.h"
#import "NFKServedOpenAI.h"
#import "NFKInferenceWireCoding.h"
#import "NFKInferenceServerPrivate.h"
#import "NFKServedRequestChecks.h"
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKInferenceTool.h>
#import <InferKit/NFKToolMessages.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKErrors.h>

NSString * const NFKServedGeminiBasePath = @"/v1beta";

static NSDictionary<NSString *, NSString *> *NFKServedGeminiFields(void)
{
	return @{ @"contents": @"array", @"systemInstruction": @"object|string", @"tools": @"array", @"toolConfig": @"object",
			  @"generationConfig": @"object", @"safetySettings": @"array", @"cachedContent": @"string",
			  @"generateContentRequest": @"object" };
}

static NSString *NFKServedGeminiJSONText(id object)
{
	NSData *data = [NSJSONSerialization isValidJSONObject:object]
		? [NSJSONSerialization dataWithJSONObject:object options:NSJSONWritingSortedKeys error:NULL] : nil;
	return data != nil ? [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] : @"{}";
}

static NSString *NFKServedGeminiIdentifier(NSString *prefix)
{
	return [prefix stringByAppendingString:[[NSUUID UUID].UUIDString stringByReplacingOccurrencesOfString:@"-" withString:@""].lowercaseString];
}

/*! A schema with its type names lowercased, as Gemini's OpenAPI subset writes them in capitals (OBJECT,
	STRING) where JSON Schema writes them in lowercase. */
static id NFKServedGeminiSchema(id schema)
{
	if ([schema isKindOfClass:NSArray.class]) {
		NSMutableArray *items = [NSMutableArray array];
		for (id item in schema) {
			[items addObject:NFKServedGeminiSchema(item)];
		}
		return items;
	}
	if (![schema isKindOfClass:NSDictionary.class]) {
		return schema;
	}
	NSMutableDictionary *converted = [NSMutableDictionary dictionary];
	for (NSString *key in schema) {
		id value = schema[key];
		if ([key isEqualToString:@"type"] && [value isKindOfClass:NSString.class]) {
			converted[key] = [value lowercaseString];
		} else if ([key isEqualToString:@"type"] && [value isKindOfClass:NSArray.class]) {
			converted[key] = [value valueForKey:@"lowercaseString"];
		} else {
			converted[key] = NFKServedGeminiSchema(value);
		}
	}
	return converted;
}

@implementation NFKServedGemini

#pragma mark Routes

+ (void)serveRequest:(NFKHTTPRequest *)request path:(NSString *)path response:(NFKHTTPResponse *)response
			  server:(NFKInferenceServer *)server
{
	NSString *models = [NFKServedGeminiBasePath stringByAppendingString:@"/models"];
	if ([path isEqualToString:models]) {
		if ([self requireMethod:@"GET" of:request response:response]) {
			[self serveModelListWithResponse:response server:server];
		}
		return;
	}
	NSString *name = [path hasPrefix:[models stringByAppendingString:@"/"]] ? [path substringFromIndex:models.length + 1] : nil;
	if (name.length == 0) {
		[self sendStatus:404 message:[NSString stringWithFormat:@"no route %@ %@", request.method, request.path] response:response];
		return;
	}
	// The method follows the model's last colon: models/{model}:generateContent.
	NSRange colon = [name rangeOfString:@":" options:NSBackwardsSearch];
	if (colon.location == NSNotFound) {
		if ([self requireMethod:@"GET" of:request response:response]) {
			[self serveModelNamed:name response:response server:server];
		}
		return;
	}
	NSString *method = [name substringFromIndex:NSMaxRange(colon)];
	NSString *model = [name substringToIndex:colon.location];
	if (![@[ @"generateContent", @"streamGenerateContent", @"countTokens" ] containsObject:method]) {
		[self sendStatus:404 message:[NSString stringWithFormat:@"this server does not serve %@", method] response:response];
		return;
	}
	if (![self requireMethod:@"POST" of:request response:response]) {
		return;
	}
	if ([method isEqualToString:@"countTokens"]) {
		[self serveTokenCount:request model:model response:response server:server];
	} else {
		BOOL streams = [method isEqualToString:@"streamGenerateContent"];
		[self serveGenerate:request model:model streaming:streams events:streams && [request.query[@"alt"] isEqual:@"sse"]
				   response:response server:server];
	}
}

+ (BOOL)requireMethod:(NSString *)method of:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response
{
	if ([request.method isEqualToString:method]) {
		return YES;
	}
	NSDictionary *body = @{ @"error": @{ @"code": @405, @"message": [NSString stringWithFormat:@"%@ takes %@", request.path, method],
										  @"status": @"INVALID_ARGUMENT" } };
	[response sendStatus:405 JSONObject:body headers:@{ @"Allow": method }];
	return NO;
}

+ (void)sendStatus:(NSInteger)status message:(NSString *)message response:(NFKHTTPResponse *)response
{
	[response sendStatus:status JSONObject:@{ @"error": @{ @"code": @(status), @"message": message,
														   @"status": [self statusNameForStatus:status] } } headers:nil];
}

/*! The model a body asks for: the path's name, with a "models/" prefix the SDKs may repeat taken off. */
+ (NSString *)modelNameOf:(NSString *)name
{
	return [name hasPrefix:@"models/"] ? [name substringFromIndex:7] : name;
}

+ (void)serveGenerate:(NFKHTTPRequest *)request
				model:(NSString *)name
			streaming:(BOOL)streaming
			   events:(BOOL)events
			 response:(NFKHTTPResponse *)response
			   server:(NFKInferenceServer *)server
{
	NSDictionary *body = [server JSONBodyOf:request response:response wire:NFKServedWireGemini fields:NFKServedGeminiFields()];
	if (body == nil) {
		return;
	}
	NSError *error = nil;
	BOOL includeThoughts = NO;
	NFKServedTarget *target = [server modelNamed:[self modelNameOf:name] error:&error];
	NFKInferenceRequest *inference = target != nil ? [self requestFromBody:body includeThoughts:&includeThoughts error:&error] : nil;
	if (inference == nil) {
		[server sendError:error response:response wire:NFKServedWireGemini];
		return;
	}
	NSString *identifier = NFKServedGeminiIdentifier(@"");
	NSString *modelName = target.name;
	NFKServedRun *run = [[NFKServedRun alloc] init];
	run.request = inference;
	if (!events) {
		run.completionHandler = ^(NFKInferenceResult *result, NSError *runError) {
			if (result == nil) {
				[server sendError:runError response:response wire:NFKServedWireGemini];
				return;
			}
			NSDictionary *reply = [self responseForResult:result request:inference model:modelName identifier:identifier
										  includeThoughts:includeThoughts];
			[response sendStatus:200 JSONObject:streaming ? @[ reply ] : reply headers:nil];
		};
		[server enqueue:run on:target request:request response:response];
		return;
	}
	NFKServedGeminiStream *stream = [[NFKServedGeminiStream alloc] initWithRequest:inference model:modelName identifier:identifier
																  includeThoughts:includeThoughts response:response];
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
				[server sendError:runError response:response wire:NFKServedWireGemini];
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
	[server enqueue:run on:target request:request response:response];
}

/*! countTokens takes contents alone or a whole generateContentRequest. */
+ (void)serveTokenCount:(NFKHTTPRequest *)request model:(NSString *)name response:(NFKHTTPResponse *)response
				 server:(NFKInferenceServer *)server
{
	NSDictionary *body = [server JSONBodyOf:request response:response wire:NFKServedWireGemini fields:NFKServedGeminiFields()];
	if (body == nil) {
		return;
	}
	NSDictionary *generate = [body[@"generateContentRequest"] isKindOfClass:NSDictionary.class] ? body[@"generateContentRequest"] : body;
	NSError *error = NFKServedFieldTypeError(generate, NFKServedGeminiFields(), @"generateContentRequest");
	NFKServedTarget *target = error == nil ? [server modelNamed:[self modelNameOf:name] error:&error] : nil;
	NFKInferenceRequest *inference = target != nil ? [self requestFromBody:generate includeThoughts:NULL error:&error] : nil;
	if (inference == nil) {
		[server sendError:error response:response wire:NFKServedWireGemini];
		return;
	}
	NFKServedRun *run = [NFKServedRun tokenCountRunForRequest:inference modelName:target.name];
	run.completionHandler = ^(NFKInferenceResult *result, NSError *runError) {
		NSNumber *count = result != nil ? [result outputForKey:NFKOutputUsage][NFKUsageInputTokens] : nil;
		if (count == nil) {
			[server sendError:runError response:response wire:NFKServedWireGemini];
			return;
		}
		[response sendStatus:200 JSONObject:@{ @"totalTokens": count } headers:nil];
	};
	[server enqueue:run on:target request:request response:response];
}

+ (void)serveModelListWithResponse:(NFKHTTPResponse *)response server:(NFKInferenceServer *)server
{
	NSMutableArray *entries = [NSMutableArray array];
	for (NSString *name in [server.modelNames arrayByAddingObjectsFromArray:server.aliasNames]) {
		[entries addObject:[self modelEntryForName:name]];
	}
	[entries sortUsingDescriptors:@[ [NSSortDescriptor sortDescriptorWithKey:@"name" ascending:YES] ]];
	[response sendStatus:200 JSONObject:@{ @"models": entries } headers:nil];
}

+ (void)serveModelNamed:(NSString *)name response:(NFKHTTPResponse *)response server:(NFKInferenceServer *)server
{
	NSError *error = nil;
	NFKServedTarget *target = [server modelNamed:[self modelNameOf:name] error:&error];
	if (target == nil) {
		[server sendError:error response:response wire:NFKServedWireGemini];
		return;
	}
	[response sendStatus:200 JSONObject:[self modelEntryForName:target.name] headers:nil];
}

+ (NSDictionary<NSString *, id> *)modelEntryForName:(NSString *)name
{
	return @{ @"name": [@"models/" stringByAppendingString:name], @"baseModelId": name, @"version": @"1", @"displayName": name,
			  @"supportedGenerationMethods": @[ @"generateContent", @"streamGenerateContent", @"countTokens" ] };
}

#pragma mark Request

+ (nullable NFKInferenceRequest *)requestFromBody:(NSDictionary *)body
								  includeThoughts:(nullable BOOL *)outIncludeThoughts
											error:(NSError * _Nullable *)outError
{
	NSArray *contents = [body[@"contents"] isKindOfClass:NSArray.class] ? body[@"contents"] : nil;
	if (contents.count == 0) {
		[NFKServedOpenAI setError:outError code:kNFKError_InferenceMissingInput reason:@"the request carries no contents"];
		return nil;
	}
	NSMutableArray<NSDictionary *> *messages = [NSMutableArray array];
	NSString *system = [self textOfInstruction:body[@"systemInstruction"]];
	if (system.length > 0) {
		[messages addObject:@{ @"role": @"system", @"content": system }];
	}
	NSMutableArray *images = [NSMutableArray array];
	NSMutableArray *documents = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *pendingCalls = [NSMutableArray array];
	for (NSUInteger index = 0; index < contents.count; index++) {
		if (![self readContent:contents[index] index:index into:messages images:images documents:documents
				  pendingCalls:pendingCalls error:outError]) {
			return nil;
		}
	}
	BOOL includeThoughts = NO;
	NSDictionary *parameters = [self parametersFromBody:body includeThoughts:&includeThoughts error:outError];
	if (parameters == nil) {
		return nil;
	}
	if (outIncludeThoughts != NULL) {
		*outIncludeThoughts = includeThoughts;
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

/*! systemInstruction as text: a string, or its parts' text joined by a blank line. */
+ (nullable NSString *)textOfInstruction:(nullable id)instruction
{
	if ([instruction isKindOfClass:NSString.class]) {
		return instruction;
	}
	NSMutableArray<NSString *> *texts = [NSMutableArray array];
	NSArray *parts = [instruction isKindOfClass:NSDictionary.class] && [instruction[@"parts"] isKindOfClass:NSArray.class] ? instruction[@"parts"] : @[];
	for (NSDictionary *part in parts) {
		if ([part isKindOfClass:NSDictionary.class] && [part[@"text"] isKindOfClass:NSString.class]) {
			[texts addObject:part[@"text"]];
		}
	}
	return [texts componentsJoinedByString:@"\n\n"];
}

// A user turn's functionResponse parts become tool turns ahead of its own text; a model turn's
// functionCall parts become its tool_calls. pendingCalls holds the calls no response has answered yet,
// so a response without an id answers the earliest call of its name.
+ (BOOL)readContent:(id)content
			  index:(NSUInteger)index
			   into:(NSMutableArray<NSDictionary *> *)messages
			 images:(NSMutableArray *)images
		  documents:(NSMutableArray *)documents
	   pendingCalls:(NSMutableArray<NSDictionary *> *)pendingCalls
			  error:(NSError * _Nullable *)outError
{
	NSString *role = [content isKindOfClass:NSDictionary.class] && [content[@"role"] isKindOfClass:NSString.class] ? content[@"role"] : @"user";
	NSArray *parts = [content isKindOfClass:NSDictionary.class] && [content[@"parts"] isKindOfClass:NSArray.class] ? content[@"parts"] : nil;
	if (parts == nil || !([role isEqualToString:@"user"] || [role isEqualToString:@"model"] || [role isEqualToString:@"function"])) {
		return [NFKServedOpenAI setError:outError code:kNFKError_InferenceMissingInput
								  reason:[NSString stringWithFormat:@"contents[%lu] must be an object with parts whose role is user or model",
										  (unsigned long)index]];
	}
	BOOL fromModel = [role isEqualToString:@"model"];
	NSMutableArray<NSString *> *texts = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *calls = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *toolTurns = [NSMutableArray array];
	for (NSDictionary *part in parts) {
		if (![part isKindOfClass:NSDictionary.class]) {
			return [NFKServedOpenAI setError:outError code:kNFKError_InferenceMissingInput
									  reason:[NSString stringWithFormat:@"contents[%lu] has a part that is not an object", (unsigned long)index]];
		}
		if ([part[@"text"] isKindOfClass:NSString.class]) {
			// A thought carries a signature only Google can verify, so it is not replayed.
			if (![part[@"thought"] isEqual:@YES]) {
				[texts addObject:part[@"text"]];
			}
		} else if ([part[@"functionCall"] isKindOfClass:NSDictionary.class]) {
			NSDictionary *call = [self callForFunctionCall:part[@"functionCall"]];
			[calls addObject:call];
			[pendingCalls addObject:call];
		} else if ([part[@"functionResponse"] isKindOfClass:NSDictionary.class]) {
			NSDictionary *turn = [self toolTurnForFunctionResponse:part[@"functionResponse"] pendingCalls:pendingCalls error:outError];
			if (turn == nil) {
				return NO;
			}
			[toolTurns addObject:turn];
		} else if ([part[@"inlineData"] isKindOfClass:NSDictionary.class]) {
			if (![self readInlineData:part[@"inlineData"] images:images documents:documents error:outError]) {
				return NO;
			}
		} else if (part[@"fileData"] != nil) {
			return [NFKServedOpenAI setError:outError code:kNFKError_InferenceUnsupported
									  reason:@"media must be inline data; this server fetches no file URI"];
		} else if (part[@"thoughtSignature"] == nil && part[@"executableCode"] == nil && part[@"codeExecutionResult"] == nil) {
			return [NFKServedOpenAI setError:outError code:kNFKError_InferenceUnsupported
									  reason:[NSString stringWithFormat:@"contents[%lu] has a part this server does not serve", (unsigned long)index]];
		}
	}
	NSString *text = [texts componentsJoinedByString:@""];
	if (fromModel) {
		[messages addObject:[NFKToolMessages assistantMessageWithText:text toolCalls:calls record:nil format:nil source:nil]];
		return YES;
	}
	[messages addObjectsFromArray:toolTurns];
	if (texts.count > 0 || toolTurns.count == 0) {
		[messages addObject:@{ @"role": @"user", @"content": text }];
	}
	return YES;
}

+ (NSDictionary *)callForFunctionCall:(NSDictionary *)functionCall
{
	NSString *name = [functionCall[@"name"] isKindOfClass:NSString.class] ? functionCall[@"name"] : @"";
	NSDictionary *arguments = [functionCall[@"args"] isKindOfClass:NSDictionary.class] ? functionCall[@"args"] : @{};
	NSString *identifier = [functionCall[@"id"] isKindOfClass:NSString.class] && [functionCall[@"id"] length] > 0
		? functionCall[@"id"] : NFKServedGeminiIdentifier(@"call_");
	return @{ @"id": identifier, @"name": name, @"arguments": arguments, @"argumentsJSON": NFKServedGeminiJSONText(arguments) };
}

/*! The tool turn a functionResponse makes: its response's output as the text, or its error as a failed
	call's text, or the whole response as JSON. */
+ (nullable NSDictionary *)toolTurnForFunctionResponse:(NSDictionary *)functionResponse
										  pendingCalls:(NSMutableArray<NSDictionary *> *)pendingCalls
												 error:(NSError * _Nullable *)outError
{
	NSString *name = [functionResponse[@"name"] isKindOfClass:NSString.class] ? functionResponse[@"name"] : nil;
	NSString *identifier = [functionResponse[@"id"] isKindOfClass:NSString.class] ? functionResponse[@"id"] : nil;
	NSUInteger answered = [pendingCalls indexOfObjectPassingTest:^BOOL(NSDictionary *call, NSUInteger index, BOOL *stop) {
		return identifier.length > 0 ? [call[@"id"] isEqualToString:identifier] : [call[@"name"] isEqualToString:name ?: @""];
	}];
	if (answered == NSNotFound) {
		[NFKServedOpenAI setError:outError code:kNFKError_InferenceMissingInput
						   reason:@"a functionResponse answers no earlier functionCall of its name"];
		return nil;
	}
	NSDictionary *call = pendingCalls[answered];
	[pendingCalls removeObjectAtIndex:answered];
	NSDictionary *response = [functionResponse[@"response"] isKindOfClass:NSDictionary.class] ? functionResponse[@"response"] : @{};
	BOOL failed = response[@"error"] != nil && response[@"output"] == nil;
	id value = failed ? response[@"error"] : (response.count == 1 && response[@"output"] != nil ? response[@"output"] : response);
	NSString *text = [value isKindOfClass:NSString.class] ? value : NFKServedGeminiJSONText(value);
	NSMutableArray<NSData *> *pngs = [NSMutableArray array];
	for (NSDictionary *part in [functionResponse[@"parts"] isKindOfClass:NSArray.class] ? functionResponse[@"parts"] : @[]) {
		NSDictionary *inlineData = [part isKindOfClass:NSDictionary.class] ? part[@"inlineData"] : nil;
		NSData *bytes = [inlineData[@"data"] isKindOfClass:NSString.class]
			? [[NSData alloc] initWithBase64EncodedString:inlineData[@"data"] options:NSDataBase64DecodingIgnoreUnknownCharacters] : nil;
		id image = bytes != nil ? (__bridge_transfer id)[NFKImageCoding pixelBufferWithImageData:bytes] : nil;
		NSData *png = image != nil ? [NFKImageCoding PNGDataForImage:image] : nil;
		if (png != nil) {
			[pngs addObject:png];
		}
	}
	NFKToolResult *result = [[NFKToolResult alloc] initWithText:text outputs:nil imagePNGs:pngs isError:failed];
	return [NFKToolMessages messageForToolCallIdentifier:call[@"id"] name:call[@"name"] result:result];
}

/*! An inline part: an image to the image inputs, a PDF's bytes or plain text to the documents. */
+ (BOOL)readInlineData:(NSDictionary *)inlineData images:(NSMutableArray *)images documents:(NSMutableArray *)documents
				 error:(NSError * _Nullable *)outError
{
	NSString *mimeType = [inlineData[@"mimeType"] isKindOfClass:NSString.class] ? inlineData[@"mimeType"] : @"";
	NSData *bytes = [inlineData[@"data"] isKindOfClass:NSString.class]
		? [[NSData alloc] initWithBase64EncodedString:inlineData[@"data"] options:NSDataBase64DecodingIgnoreUnknownCharacters] : nil;
	if (bytes == nil) {
		return [NFKServedOpenAI setError:outError code:kNFKError_InferenceUnsupported reason:@"inlineData carries no base64 data"];
	}
	if ([mimeType hasPrefix:@"image/"]) {
		CVPixelBufferRef buffer = [NFKImageCoding pixelBufferWithImageData:bytes];
		if (buffer == NULL) {
			return [NFKServedOpenAI setError:outError code:kNFKError_InferenceUnsupported
									  reason:@"an inlineData image is not an image this server can decode"];
		}
		[images addObject:(__bridge_transfer id)buffer];
		return YES;
	}
	if ([mimeType isEqualToString:@"application/pdf"]) {
		[documents addObject:bytes];
		return YES;
	}
	if ([mimeType hasPrefix:@"text/"]) {
		[documents addObject:[[NSString alloc] initWithData:bytes encoding:NSUTF8StringEncoding] ?: @""];
		return YES;
	}
	return [NFKServedOpenAI setError:outError code:kNFKError_InferenceUnsupported
							  reason:[NSString stringWithFormat:@"inlineData of type %@ is not served", mimeType.length > 0 ? mimeType : @"(none)"]];
}

#pragma mark Parameters

+ (nullable NSDictionary<NSString *, id> *)parametersFromBody:(NSDictionary *)body
											  includeThoughts:(BOOL *)outIncludeThoughts
														error:(NSError * _Nullable *)outError
{
	NSMutableDictionary<NSString *, id> *parameters = [NSMutableDictionary dictionary];
	NSDictionary *generation = [body[@"generationConfig"] isKindOfClass:NSDictionary.class] ? body[@"generationConfig"] : @{};
	NSError *typeError = NFKServedFieldTypeError(generation, @{ @"maxOutputTokens": @"integer", @"temperature": @"number",
																@"topP": @"number", @"topK": @"integer", @"seed": @"integer",
																@"stopSequences": @"array", @"responseMimeType": @"string",
																@"responseSchema": @"object", @"responseJsonSchema": @"object",
																@"thinkingConfig": @"object" }, @"generationConfig");
	if (typeError != nil) {
		if (outError != NULL) {
			*outError = typeError;
		}
		return nil;
	}
	NSDictionary<NSString *, NSString *> *renamed = @{ @"maxOutputTokens": NFKParameterMaxTokens, @"temperature": NFKParameterTemperature,
													   @"topP": NFKParameterTopP, @"topK": NFKParameterTopK, @"seed": NFKParameterSeed };
	for (NSString *key in renamed) {
		if ([generation[key] isKindOfClass:NSNumber.class]) {
			parameters[renamed[key]] = generation[key];
		}
	}
	if ([generation[@"stopSequences"] isKindOfClass:NSArray.class]) {
		parameters[NFKParameterStopSequences] = generation[@"stopSequences"];
	}
	if ([generation[@"responseMimeType"] isEqual:@"application/json"]) {
		id schema = generation[@"responseJsonSchema"] ?: NFKServedGeminiSchema(generation[@"responseSchema"]);
		if ([schema isKindOfClass:NSDictionary.class]) {
			parameters[NFKParameterJSONSchema] = schema;
		} else {
			parameters[NFKParameterOutputFormat] = @"json";
		}
	}
	NSDictionary *thinking = [generation[@"thinkingConfig"] isKindOfClass:NSDictionary.class] ? generation[@"thinkingConfig"] : @{};
	*outIncludeThoughts = [thinking[@"includeThoughts"] isEqual:@YES];
	NSString *effort = [self effortForThinking:thinking];
	if (effort != nil) {
		parameters[NFKParameterReasoningEffort] = effort;
	}
	NSArray *tools = [self toolsOfBody:body];
	if (tools.count > 0) {
		parameters[NFKParameterTools] = tools;
	}
	NSDictionary *toolConfig = [body[@"toolConfig"] isKindOfClass:NSDictionary.class] ? body[@"toolConfig"] : nil;
	NSDictionary *calling = [toolConfig[@"functionCallingConfig"] isKindOfClass:NSDictionary.class] ? toolConfig[@"functionCallingConfig"] : nil;
	NSDictionary<NSString *, NSString *> *modes = @{ @"AUTO": @"auto", @"ANY": @"required", @"NONE": @"none", @"VALIDATED": @"auto" };
	NSString *mode = [calling[@"mode"] isKindOfClass:NSString.class] ? modes[[calling[@"mode"] uppercaseString]] : nil;
	if (mode != nil) {
		parameters[@"tool_choice"] = mode;
	}
	return parameters;
}

// A budget passes through as its token count in a string, which the contract's effort key takes; 0 turns
// thinking off and -1 lets the model decide, which the deep effort stands for. A level names the effort.
+ (nullable NSString *)effortForThinking:(NSDictionary *)thinking
{
	NSNumber *budget = [thinking[@"thinkingBudget"] isKindOfClass:NSNumber.class] ? thinking[@"thinkingBudget"] : nil;
	if (budget != nil) {
		return budget.integerValue > 0 ? budget.stringValue : (budget.integerValue < 0 ? NFKReasoningEffortDeep : nil);
	}
	NSString *level = [thinking[@"thinkingLevel"] isKindOfClass:NSString.class] ? [thinking[@"thinkingLevel"] uppercaseString] : nil;
	NSDictionary<NSString *, NSString *> *levels = @{ @"MINIMAL": NFKReasoningEffortLight, @"LOW": NFKReasoningEffortLight,
													  @"MEDIUM": NFKReasoningEffortModerate, @"HIGH": NFKReasoningEffortDeep };
	return level != nil ? levels[level] : nil;
}

/*! The function declarations across the body's tools in the contract's shape. Search, code execution, and
	the other tools that run at Google carry no declaration and are left out. */
+ (NSArray<NSDictionary *> *)toolsOfBody:(NSDictionary *)body
{
	NSMutableArray<NSDictionary *> *tools = [NSMutableArray array];
	for (NSDictionary *tool in [body[@"tools"] isKindOfClass:NSArray.class] ? body[@"tools"] : @[]) {
		NSArray *declarations = [tool isKindOfClass:NSDictionary.class] && [tool[@"functionDeclarations"] isKindOfClass:NSArray.class]
			? tool[@"functionDeclarations"] : @[];
		for (NSDictionary *declaration in declarations) {
			if (![declaration isKindOfClass:NSDictionary.class] || ![declaration[@"name"] isKindOfClass:NSString.class]) {
				continue;
			}
			id schema = declaration[@"parametersJsonSchema"] ?: NFKServedGeminiSchema(declaration[@"parameters"]);
			NSMutableDictionary *contract = [NSMutableDictionary dictionaryWithObject:declaration[@"name"] forKey:@"name"];
			contract[@"parameters"] = [schema isKindOfClass:NSDictionary.class] ? schema : @{ @"type": @"object", @"properties": @{} };
			contract[@"description"] = [declaration[@"description"] isKindOfClass:NSString.class] ? declaration[@"description"] : nil;
			[tools addObject:contract];
		}
	}
	return tools;
}

#pragma mark Reply

+ (nullable NSString *)reasoningForResult:(nullable NFKInferenceResult *)result
{
	NSString *reasoning = [result outputForKey:NFKOutputReasoning];
	return [reasoning isKindOfClass:NSString.class] ? reasoning : nil;
}

/*! The functionCall parts a result calls for, each with its id, name, and args. */
+ (NSArray<NSDictionary *> *)functionCallPartsForResult:(NFKInferenceResult *)result
{
	NSMutableArray<NSDictionary *> *parts = [NSMutableArray array];
	for (NSDictionary *call in result.toolCalls ?: @[]) {
		if (![call isKindOfClass:NSDictionary.class] || ![call[@"name"] isKindOfClass:NSString.class]) {
			continue;
		}
		id arguments = call[@"arguments"];
		if (![arguments isKindOfClass:NSDictionary.class] && [call[@"argumentsJSON"] isKindOfClass:NSString.class]) {
			arguments = [NSJSONSerialization JSONObjectWithData:[call[@"argumentsJSON"] dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
		}
		NSString *identifier = [call[@"id"] isKindOfClass:NSString.class] && [call[@"id"] length] > 0 ? call[@"id"] : NFKServedGeminiIdentifier(@"call_");
		[parts addObject:@{ @"functionCall": @{ @"id": identifier, @"name": call[@"name"],
												@"args": [arguments isKindOfClass:NSDictionary.class] ? arguments : @{} } }];
	}
	return parts;
}

+ (NSString *)finishReasonForResult:(NFKInferenceResult *)result request:(NFKInferenceRequest *)request
{
	NSString *reported = [result outputForKey:NFKOutputFinishReason];
	if ([reported isEqual:NFKFinishReasonLength]) {
		return @"MAX_TOKENS";
	}
	if ([reported isEqual:NFKFinishReasonContentFilter]) {
		return @"SAFETY";
	}
	if (reported == nil) {
		NSNumber *limit = [request parameterForKey:NFKParameterMaxTokens];
		NSInteger produced = NFKServedUsageCount([result outputForKey:NFKOutputUsage], NFKUsageOutputTokens);
		if ([limit isKindOfClass:NSNumber.class] && limit.integerValue > 0 && produced >= limit.integerValue) {
			return @"MAX_TOKENS";
		}
	}
	return @"STOP";
}

/*! usageMetadata. Every reply carries one, with zeros for what the backend does not report. */
+ (NSDictionary<NSString *, id> *)usageMetadataForResult:(nullable NFKInferenceResult *)result
{
	NSDictionary *usage = [[result outputForKey:NFKOutputUsage] isKindOfClass:NSDictionary.class] ? [result outputForKey:NFKOutputUsage] : @{};
	NSInteger input = NFKServedUsageCount(usage, NFKUsageInputTokens);
	NSInteger output = NFKServedUsageCount(usage, NFKUsageOutputTokens);
	NSInteger thoughts = MIN(NFKServedUsageCount(usage, NFKUsageReasoningTokens), output);
	NSMutableDictionary *metadata = [@{ @"promptTokenCount": @(input), @"candidatesTokenCount": @(output - thoughts),
										@"totalTokenCount": @(input + output) } mutableCopy];
	if (thoughts > 0) {
		metadata[@"thoughtsTokenCount"] = @(thoughts);
	}
	if ([usage[NFKUsageCachedTokens] isKindOfClass:NSNumber.class]) {
		metadata[@"cachedContentTokenCount"] = @(MIN([usage[NFKUsageCachedTokens] integerValue], input));
	}
	return metadata;
}

+ (NSDictionary<NSString *, id> *)responseForResult:(NFKInferenceResult *)result
											request:(NFKInferenceRequest *)request
											  model:(NSString *)model
										 identifier:(NSString *)identifier
									includeThoughts:(BOOL)includeThoughts
{
	NSMutableArray<NSDictionary *> *parts = [NSMutableArray array];
	NSString *reasoning = includeThoughts ? [self reasoningForResult:result] : nil;
	if (reasoning.length > 0) {
		[parts addObject:@{ @"text": reasoning, @"thought": @YES }];
	}
	NSArray<NSDictionary *> *calls = [self functionCallPartsForResult:result];
	NSString *text = [NFKServedOpenAI textForResult:result];
	if (text.length > 0 || (parts.count == 0 && calls.count == 0)) {
		[parts addObject:@{ @"text": text ?: @"" }];
	}
	[parts addObjectsFromArray:calls];
	NSDictionary *candidate = @{ @"content": @{ @"role": @"model", @"parts": parts }, @"index": @0,
								 @"finishReason": [self finishReasonForResult:result request:request] };
	return @{ @"candidates": @[ candidate ], @"usageMetadata": [self usageMetadataForResult:result],
			  @"modelVersion": model, @"responseId": identifier };
}

#pragma mark Errors

+ (NSString *)statusNameForStatus:(NSInteger)status
{
	NSDictionary<NSNumber *, NSString *> *names = @{ @400: @"INVALID_ARGUMENT", @401: @"UNAUTHENTICATED", @403: @"PERMISSION_DENIED",
													 @404: @"NOT_FOUND", @405: @"INVALID_ARGUMENT", @421: @"PERMISSION_DENIED",
													 @429: @"RESOURCE_EXHAUSTED", @501: @"UNIMPLEMENTED", @503: @"UNAVAILABLE",
													 @502: @"UNAVAILABLE", @504: @"DEADLINE_EXCEEDED" };
	return names[@(status)] ?: (status >= 500 ? @"INTERNAL" : @"INVALID_ARGUMENT");
}

+ (NSDictionary<NSString *, id> *)errorObjectForError:(NSError *)error status:(NSInteger)status
{
	NSMutableDictionary<NSString *, id> *detail = [[NFKInferenceWireCoding JSONObjectForError:error][@"error"] mutableCopy];
	[detail removeObjectForKey:@"type"];
	detail[@"code"] = @(status);
	detail[@"status"] = [self statusNameForStatus:status];
	return @{ @"error": detail };
}

@end

#pragma mark - Streaming

@interface NFKServedGeminiStream ()
@property (nonatomic, strong) NFKInferenceRequest *request;
@property (nonatomic, copy) NSString *model;
@property (nonatomic, copy) NSString *identifier;
@property (nonatomic, assign) BOOL includeThoughts;
@property (nonatomic, strong) NFKHTTPResponse *response;
@property (nonatomic, readwrite) BOOL started;
@property (nonatomic, strong) NSMutableString *sentText;
@property (nonatomic, strong) NSMutableString *sentReasoning;
@end

@implementation NFKServedGeminiStream

- (instancetype)initWithRequest:(NFKInferenceRequest *)request
						  model:(NSString *)model
					 identifier:(NSString *)identifier
				includeThoughts:(BOOL)includeThoughts
					   response:(NFKHTTPResponse *)response
{
	self = [super init];
	if (self != nil) {
		_request = request;
		_model = [model copy];
		_identifier = [identifier copy];
		_includeThoughts = includeThoughts;
		_response = response;
		_sentText = [NSMutableString string];
		_sentReasoning = [NSMutableString string];
	}
	return self;
}

- (void)send:(NSDictionary<NSString *, id> *)chunk
{
	if (!self.response.headersSent) {
		[self.response beginStreamWithStatus:200 headers:@{ @"Content-Type": @"text/event-stream", @"Cache-Control": @"no-cache" }];
	}
	self.started = YES;
	NSData *JSON = [NSJSONSerialization dataWithJSONObject:chunk options:0 error:NULL];
	[self.response sendEventData:JSON != nil ? [[NSString alloc] initWithData:JSON encoding:NSUTF8StringEncoding] : @"{}"];
}

/*! One chunk carrying parts, and the finish reason and usage when it closes the reply. */
- (void)sendParts:(NSArray<NSDictionary *> *)parts finishReason:(nullable NSString *)finishReason usage:(nullable NSDictionary *)usage
{
	NSMutableDictionary *candidate = [NSMutableDictionary dictionaryWithDictionary:@{ @"content": @{ @"role": @"model", @"parts": parts },
																					  @"index": @0 }];
	candidate[@"finishReason"] = finishReason;
	NSMutableDictionary *chunk = [NSMutableDictionary dictionaryWithDictionary:@{ @"candidates": @[ candidate ], @"modelVersion": self.model,
																				  @"responseId": self.identifier }];
	chunk[@"usageMetadata"] = usage;
	[self send:chunk];
}

/*! The parts that carry what result's reasoning and text add to what was sent. */
- (NSArray<NSDictionary *> *)partsAdvancingTo:(NFKInferenceResult *)result
{
	NSMutableArray<NSDictionary *> *parts = [NSMutableArray array];
	NSString *reasoning = self.includeThoughts ? NFKServedSuffix([NFKServedGemini reasoningForResult:result], self.sentReasoning) : nil;
	if (reasoning != nil) {
		[self.sentReasoning appendString:reasoning];
		[parts addObject:@{ @"text": reasoning, @"thought": @YES }];
	}
	NSString *text = NFKServedSuffix(result.text, self.sentText);
	if (text != nil) {
		[self.sentText appendString:text];
		[parts addObject:@{ @"text": text }];
	}
	return parts;
}

- (void)appendPartial:(NFKInferenceResult *)partial
{
	NSArray<NSDictionary *> *parts = [self partsAdvancingTo:partial];
	if (parts.count > 0) {
		[self sendParts:parts finishReason:nil usage:nil];
	}
}

- (void)finishWithResult:(NFKInferenceResult *)result
{
	NSMutableArray<NSDictionary *> *parts = [[self partsAdvancingTo:result] mutableCopy];
	NSArray<NSDictionary *> *calls = [NFKServedGemini functionCallPartsForResult:result];
	if (self.sentText.length == 0) {
		NSString *text = [NFKServedOpenAI textForResult:result];
		if (text.length > 0) {
			[self.sentText appendString:text];
			[parts addObject:@{ @"text": text }];
		}
	}
	[parts addObjectsFromArray:calls];
	if (parts.count == 0) {
		[parts addObject:@{ @"text": @"" }];
	}
	[self sendParts:parts finishReason:[NFKServedGemini finishReasonForResult:result request:self.request]
			  usage:[NFKServedGemini usageMetadataForResult:result]];
}

- (void)failWithError:(NSError *)error status:(NSInteger)status
{
	[self send:[NFKServedGemini errorObjectForError:error status:status]];
}

@end
