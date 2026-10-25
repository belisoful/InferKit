//
//  NFKMistralConversationsBackend.m
//  InferKit
//

#import <InferKit/NFKMistralConversationsBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKToolMessages.h>
#import <InferKit/NFKErrors.h>
#import "NFKRemoteMediaSupport.h"
#import "NFKToolMessageSupport.h"
#import "NFKRemoteTransportPrivate.h"
#import "NFKRemoteBatchPrivate.h"
#import "NFKJobGuard.h"

/*! The contract keys this backend translates; every other parameter goes out under its own name. */
static NSSet<NSString *> *NFKConversationsMappedParameters(void)
{
	static NSSet<NSString *> *mapped;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		mapped = [NSSet setWithArray:@[ NFKParameterTools, NFKParameterJSONSchema, NFKParameterReasoningEffort,
										NFKParameterMaxTokens, NFKParameterTemperature, NFKParameterTopP,
										NFKParameterStopSequences, NFKParameterSeed, NFKParameterPreviousResponseIdentifier,
										NFKParameterMCPServers, NFKParameterSkills, NFKParameterConversationKey,
										NFKParameterPromptCacheRetention, NFKParameterPromptCacheTTL,
										NFKParameterPromptCacheMode, NFKParameterPromptCachePrewarm ]];
	});
	return mapped;
}

static NSDictionary * _Nullable NFKConversationsDictionary(id value)
{
	return [value isKindOfClass:NSDictionary.class] ? value : nil;
}

static NSString * _Nullable NFKConversationsString(id value)
{
	return [value isKindOfClass:NSString.class] && [value length] > 0 ? value : nil;
}

/*! What a reply's entries, or a stream's events, add up to. */
@interface NFKConversationsReading : NSObject
@property (nonatomic, strong) NSMutableString *text;
@property (nonatomic, strong) NSMutableString *reasoning;
@property (nonatomic, strong) NSMutableArray<NSDictionary *> *toolCalls;
@property (nonatomic, strong) NSMutableArray<NSDictionary *> *citations;
@property (nonatomic, strong) NSMutableArray<NSDictionary *> *serverTools;
@property (nonatomic, strong) NSMutableArray<NSString *> *imageFiles;
@property (nonatomic, strong) NSMutableDictionary<NSString *, NSMutableDictionary *> *streamedCalls;
@property (nonatomic, strong) NSMutableArray<NSString *> *streamedCallOrder;
@property (nonatomic, copy, nullable) NSString *conversationIdentifier;
@property (nonatomic, copy, nullable) NSDictionary *usage;
@property (nonatomic, copy, nullable) NSString *failure;
@end

@implementation NFKConversationsReading

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_text = [NSMutableString string];
		_reasoning = [NSMutableString string];
		_toolCalls = [NSMutableArray array];
		_citations = [NSMutableArray array];
		_serverTools = [NSMutableArray array];
		_imageFiles = [NSMutableArray array];
		_streamedCalls = [NSMutableDictionary dictionary];
		_streamedCallOrder = [NSMutableArray array];
	}
	return self;
}

@end

@implementation NFKMistralConversationsBackend

@synthesize session = _session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKMistralConversationsBackend *backend = [[self alloc] init];
	backend.endpointURL = endpointURL;
	return backend;
}

+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
{
	if (![provider.identifier isEqualToString:@"mistral"]) {
		return nil;
	}
	NFKMistralConversationsBackend *backend = [self backendWithEndpointURL:[provider URLForPath:@"conversations"]];
	NSURLSession *pinned = NFKRemotePinnedSessionForProvider(provider);
	if (pinned != nil) {
		backend.session = pinned;
	}
	backend.apiKey = apiKey;
	backend.modelName = modelName;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_timeout = 600.0;
		_downloadsImages = YES;
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
	return self.endpointURL != nil && (self.modelName.length > 0 || self.agentIdentifier.length > 0);
}

- (NSString *)backendIdentifier
{
	return @"mistral-conversations";
}

- (NSSet<NSString *> *)supportedParameterKeys
{
	return [NSSet setWithArray:@[ NFKParameterTools, NFKParameterJSONSchema, NFKParameterReasoningEffort, NFKParameterMaxTokens,
								  NFKParameterTemperature, NFKParameterTopP, NFKParameterStopSequences, NFKParameterSeed,
								  NFKParameterPreviousResponseIdentifier ]];
}

- (NSSet<NSString *> *)supportedOutputKeys
{
	return [NSSet setWithArray:@[ NFKOutputText, NFKOutputReasoning, NFKOutputStructured, NFKOutputToolCalls,
								  NFKOutputAssistantMessage, NFKOutputCitations, NFKOutputServerToolResults,
								  NFKOutputImage, NFKOutputImages, NFKOutputResponseIdentifier, NFKOutputUsage,
								  NFKOutputFinishReason, NFKRemoteBackendRawKey ]];
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError * _Nullable *)outError
{
	NSURLRequest *urlRequest = [self urlRequestForRequest:request streaming:NO error:outError];
	if (urlRequest == nil) {
		return nil;
	}
	NSDictionary *reply = [self JSONForRequest:urlRequest error:outError];
	return reply != nil ? [self resultFromReply:reply request:request error:outError] : nil;
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	NSError *error = nil;
	NSURLRequest *urlRequest = [self urlRequestForRequest:request streaming:YES error:&error];
	if (urlRequest == nil) {
		[job finishWithError:error ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the conversation call failed"]];
		return job;
	}
	[job reportProgress:-1.0];
	NFKConversationsReading *reading = [[NFKConversationsReading alloc] init];
	__block BOOL finished = NO;
	void (^cancel)(void) = [self streamRequest:urlRequest lineHandler:^(NSString *line) {
		NSString *payload = [NFKRemoteTransport SSEDataForLine:line];
		NSDictionary *event = payload != nil
			? NFKConversationsDictionary([NSJSONSerialization JSONObjectWithData:[payload dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL]) : nil;
		if (finished || event == nil) {
			return;
		}
		if ([self readEvent:event into:reading]) {
			NSMutableDictionary<NSString *, id> *partial = [NSMutableDictionary dictionaryWithObject:[reading.text copy] forKey:NFKOutputText];
			if (reading.reasoning.length > 0) {
				partial[NFKOutputReasoning] = [reading.reasoning copy];
			}
			[job reportProgress:-1.0 partialResult:[NFKInferenceResult resultWithOutputs:partial]];
		}
	} completionHandler:^(NSHTTPURLResponse * _Nullable response, NSData * _Nullable errorBody, NSError * _Nullable streamError) {
		NFKJobRunGuarded(job, ^{
			if (finished || job.status == NFKInferenceJobStatusCancelled) {
				return;
			}
			finished = YES;
			NSError *failure = streamError ?: [NFKRemoteTransport errorForResponse:response data:errorBody];
			if (failure == nil && reading.failure != nil) {
				failure = [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:reading.failure];
			}
			if (failure != nil) {
				[job finishWithError:failure];
				return;
			}
			[self collectStreamedCallsOf:reading];
			[job finishWithResult:[self resultFromReading:reading request:request entries:@[] raw:@{}]];
		});
	}];
	job.cancellationHandler = cancel;
	return job;
}

#pragma mark Conversations

- (nullable NSArray<NSDictionary<NSString *, id> *> *)historyOfConversationWithIdentifier:(NSString *)identifier
																				   error:(NSError * _Nullable *)outError
{
	NSURL *url = [[self.endpointURL URLByAppendingPathComponent:identifier] URLByAppendingPathComponent:@"history"];
	NSDictionary *history = [self JSONForRequest:[self requestTo:url method:@"GET"] error:outError];
	if (history == nil) {
		return nil;
	}
	return [history[@"entries"] isKindOfClass:NSArray.class] ? history[@"entries"] : @[];
}

- (BOOL)deleteConversationWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError
{
	return [self dataForRequest:[self requestTo:[self.endpointURL URLByAppendingPathComponent:identifier] method:@"DELETE"] error:outError] != nil;
}

// The Files API sits beside the conversations path, under the same base.
- (nullable NSData *)contentsOfFileWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError
{
	NSURL *url = [[[self.endpointURL.URLByDeletingLastPathComponent URLByAppendingPathComponent:@"files"]
				   URLByAppendingPathComponent:identifier] URLByAppendingPathComponent:@"content"];
	NSMutableURLRequest *request = [self requestTo:url method:@"GET"];
	[request setValue:@"*/*" forHTTPHeaderField:@"Accept"];
	return [self dataForRequest:request error:outError];
}

#pragma mark Request

- (nullable NSURLRequest *)urlRequestForRequest:(NFKInferenceRequest *)request
										 streaming:(BOOL)streaming
											 error:(NSError * _Nullable *)outError
{
	if (self.endpointURL == nil) {
		return [self fail:outError code:kNFKError_InferenceNotReady reason:@"no endpoint URL is set"];
	}
	NSDictionary *parameters = request.parameters;
	NSString *previous = NFKConversationsString(parameters[NFKParameterPreviousResponseIdentifier]);
	if (previous == nil && !self.isReady) {
		return [self fail:outError code:kNFKError_InferenceNotReady reason:@"a new conversation needs a model or an agent"];
	}
	NFKRemoteAttachments *attachments = [NFKRemoteAttachments attachmentsForRequest:request error:outError];
	if (attachments == nil) {
		return nil;
	}
	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
	NSString *instructions = nil;
	id inputs = [self inputsForRequest:request attachments:attachments instructions:&instructions];
	if (inputs == nil) {
		return [self fail:outError code:kNFKError_InferenceMissingInput reason:@"the request carries neither a prompt nor messages"];
	}
	body[@"inputs"] = inputs;
	if (previous == nil) {
		body[self.agentIdentifier.length > 0 ? @"agent_id" : @"model"] = self.agentIdentifier.length > 0 ? self.agentIdentifier : self.modelName;
		body[@"instructions"] = instructions;
		NSArray *tools = [self toolsForParameters:parameters];
		if (tools.count > 0) {
			body[@"tools"] = tools;
		}
	}
	NSDictionary *completionArgs = [self completionArgsForParameters:parameters];
	if (completionArgs.count > 0) {
		body[@"completion_args"] = completionArgs;
	}
	NSSet<NSString *> *mapped = NFKConversationsMappedParameters();
	for (NSString *key in parameters) {
		if (![mapped containsObject:key]) {
			body[key] = parameters[key];
		}
	}
	if (streaming) {
		body[@"stream"] = @YES;
	}
	NSURL *url = self.endpointURL;
	if (previous != nil) {
		url = [url URLByAppendingPathComponent:previous];
		if (body[@"from_entry_id"] != nil) {
			url = [url URLByAppendingPathComponent:@"restart"];
		}
	}
	NSError *encodeError = nil;
	NSData *payload = [NSJSONSerialization isValidJSONObject:body] ? [NSJSONSerialization dataWithJSONObject:body options:0 error:&encodeError] : nil;
	if (payload == nil) {
		return [self fail:outError code:kNFKError_InferenceMissingInput reason:@"the parameters are not JSON-serializable"];
	}
	NSMutableURLRequest *urlRequest = [self requestTo:url method:@"POST"];
	urlRequest.HTTPBody = payload;
	[urlRequest setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	if (streaming) {
		[urlRequest setValue:@"text/event-stream" forHTTPHeaderField:@"Accept"];
	}
	return urlRequest;
}

// A lone prompt goes as a string. Otherwise every turn is an entry: a system turn becomes the
// instructions, a tool answer a function.result, an assistant turn this endpoint wrote replays its own
// entries, another assistant turn's calls become function.call entries, and the images ride on the last
// user turn as image_url chunks.
- (nullable id)inputsForRequest:(NFKInferenceRequest *)request
					attachments:(NFKRemoteAttachments *)attachments
				   instructions:(NSString * _Nullable * _Nonnull)outInstructions
{
	NSArray *messages = request.messages;
	if (messages.count == 0 && request.prompt.length > 0) {
		if (attachments.imagePNGs.count == 0) {
			return request.prompt;
		}
		messages = @[ @{ @"role": @"user", @"content": request.prompt } ];
	}
	NSMutableArray *entries = [NSMutableArray array];
	for (NSDictionary *message in messages) {
		if (![message isKindOfClass:NSDictionary.class]) {
			continue;
		}
		if ([message[@"role"] isEqual:@"system"] && [message[@"content"] isKindOfClass:NSString.class]) {
			*outInstructions = message[@"content"];
			continue;
		}
		if (NFKMessageIsToolResult(message)) {
			NSString *text = NFKToolMessageText(message);
			[entries addObject:@{ @"type": @"function.result", @"tool_call_id": NFKToolMessageCallIdentifier(message),
								  @"result": NFKToolMessageIsError(message) ? [@"Error: " stringByAppendingString:text] : text }];
			continue;
		}
		id recorded = NFKMessageProviderTurn(message, NFKProviderTurnFormatMistralConversations, self.endpointURL.absoluteString);
		if ([recorded isKindOfClass:NSArray.class]) {
			[entries addObjectsFromArray:recorded];
			continue;
		}
		NSString *role = [message[@"role"] isEqual:@"assistant"] ? @"assistant" : @"user";
		id content = message[@"content"];
		if ([content isKindOfClass:NSString.class] && [content length] > 0) {
			[entries addObject:@{ @"type": @"message.input", @"role": role, @"content": content }];
		} else if ([content isKindOfClass:NSArray.class]) {
			[entries addObject:@{ @"type": @"message.input", @"role": role, @"content": content }];
		}
		for (NSDictionary *call in [role isEqualToString:@"assistant"] ? [NFKToolMessages toolCallsOfAssistantMessage:message] : @[]) {
			[entries addObject:@{ @"type": @"function.call", @"tool_call_id": call[@"id"] ?: @"", @"name": call[@"name"] ?: @"",
								  @"arguments": call[@"argumentsJSON"] ?: @"{}" }];
		}
	}
	if (entries.count == 0) {
		return nil;
	}
	if (attachments.imagePNGs.count > 0) {
		[self attachImages:attachments.imagePNGs toLastUserEntryOf:entries];
	}
	return entries;
}

- (void)attachImages:(NSArray<NSData *> *)images toLastUserEntryOf:(NSMutableArray *)entries
{
	for (NSInteger index = (NSInteger)entries.count - 1; index >= 0; index--) {
		NSDictionary *entry = entries[index];
		if (![entry[@"type"] isEqual:@"message.input"] || ![entry[@"role"] isEqual:@"user"]) {
			continue;
		}
		id content = entry[@"content"];
		NSMutableArray *chunks = [content isKindOfClass:NSArray.class] ? [content mutableCopy]
							   : [NSMutableArray arrayWithObject:@{ @"type": @"text", @"text": content ?: @"" }];
		for (NSData *png in images) {
			[chunks addObject:@{ @"type": @"image_url",
								 @"image_url": [@"data:image/png;base64," stringByAppendingString:[png base64EncodedStringWithOptions:0]] }];
		}
		NSMutableDictionary *withImages = [entry mutableCopy];
		withImages[@"content"] = chunks;
		entries[(NSUInteger)index] = withImages;
		return;
	}
}

// The contract's {name, description, parameters} becomes a function tool; an entry with a type is the
// service's own and passes through.
- (NSArray<NSDictionary *> *)toolsForParameters:(NSDictionary *)parameters
{
	NSArray *declared = [parameters[NFKParameterTools] isKindOfClass:NSArray.class] ? parameters[NFKParameterTools] : @[];
	NSMutableArray<NSDictionary *> *tools = [NSMutableArray array];
	for (NSDictionary *tool in declared) {
		if (![tool isKindOfClass:NSDictionary.class]) {
			continue;
		}
		if (tool[@"type"] != nil) {
			[tools addObject:tool];
			continue;
		}
		NSMutableDictionary *function = [NSMutableDictionary dictionaryWithObject:tool[@"name"] ?: @"" forKey:@"name"];
		function[@"description"] = tool[@"description"];
		function[@"parameters"] = tool[@"parameters"] ?: @{ @"type": @"object", @"properties": @{} };
		[tools addObject:@{ @"type": @"function", @"function": function }];
	}
	return tools;
}

- (NSDictionary<NSString *, id> *)completionArgsForParameters:(NSDictionary *)parameters
{
	NSMutableDictionary<NSString *, id> *args = [NSMutableDictionary dictionary];
	args[@"temperature"] = parameters[NFKParameterTemperature];
	args[@"top_p"] = parameters[NFKParameterTopP];
	args[@"max_tokens"] = parameters[NFKParameterMaxTokens];
	args[@"stop"] = parameters[NFKParameterStopSequences];
	args[@"random_seed"] = parameters[NFKParameterSeed];
	NSString *effort = NFKConversationsString(parameters[NFKParameterReasoningEffort]);
	if (effort != nil) {
		NSDictionary<NSString *, NSString *> *levels = @{ NFKReasoningEffortLight: @"low", NFKReasoningEffortModerate: @"medium",
														  NFKReasoningEffortDeep: @"high" };
		args[@"reasoning_effort"] = levels[effort] ?: effort;
	}
	NSDictionary *schema = NFKConversationsDictionary(parameters[NFKParameterJSONSchema]);
	if (schema != nil) {
		args[@"response_format"] = @{ @"type": @"json_schema", @"json_schema": @{ @"name": @"response", @"schema": schema } };
	}
	return args;
}

#pragma mark Reply

- (nullable NFKInferenceResult *)resultFromReply:(NSDictionary *)reply request:(NFKInferenceRequest *)request error:(NSError * _Nullable *)outError
{
	NSArray *entries = [reply[@"outputs"] isKindOfClass:NSArray.class] ? reply[@"outputs"] : nil;
	if (entries == nil) {
		return [self fail:outError code:kNFKError_InferenceBackendFailure reason:@"the reply carries no outputs"];
	}
	NFKConversationsReading *reading = [[NFKConversationsReading alloc] init];
	reading.conversationIdentifier = NFKConversationsString(reply[@"conversation_id"]);
	reading.usage = NFKConversationsDictionary(reply[@"usage"]);
	for (NSDictionary *entry in entries) {
		[self readEntry:NFKConversationsDictionary(entry) ?: @{} into:reading];
	}
	return [self resultFromReading:reading request:request entries:entries raw:reply];
}

- (void)readEntry:(NSDictionary *)entry into:(NFKConversationsReading *)reading
{
	NSString *type = NFKConversationsString(entry[@"type"]) ?: @"";
	if ([type isEqualToString:@"message.output"]) {
		[self readContent:entry[@"content"] into:reading];
	} else if ([type isEqualToString:@"function.call"]) {
		[reading.toolCalls addObject:[self toolCallWithIdentifier:entry[@"tool_call_id"] name:entry[@"name"] arguments:entry[@"arguments"]]];
	} else if ([type isEqualToString:@"tool.execution"]) {
		NSMutableDictionary *ran = [NSMutableDictionary dictionaryWithObject:entry forKey:@"raw"];
		ran[@"name"] = entry[@"name"] ?: type;
		ran[@"input"] = [self parsedArguments:entry[@"arguments"]] ?: entry[@"arguments"];
		ran[@"output"] = entry[@"info"];
		[reading.serverTools addObject:ran];
	} else if ([type isEqualToString:@"agent.handoff"]) {
		NSMutableDictionary *handoff = [NSMutableDictionary dictionaryWithObject:entry forKey:@"raw"];
		handoff[@"name"] = type;
		handoff[@"input"] = entry[@"previous_agent_name"];
		handoff[@"output"] = entry[@"next_agent_name"];
		[reading.serverTools addObject:handoff];
	}
}

/*! A message's content: a string, or chunks of text, thinking, tool references (citations), and the
	files a tool wrote. */
- (void)readContent:(id)content into:(NFKConversationsReading *)reading
{
	if ([content isKindOfClass:NSString.class]) {
		[reading.text appendString:content];
		return;
	}
	NSArray *chunks = [content isKindOfClass:NSArray.class] ? content : (NFKConversationsDictionary(content) != nil ? @[ content ] : @[]);
	for (NSDictionary *chunk in chunks) {
		if (![chunk isKindOfClass:NSDictionary.class]) {
			continue;
		}
		NSString *type = NFKConversationsString(chunk[@"type"]) ?: @"";
		if ([type isEqualToString:@"text"] && [chunk[@"text"] isKindOfClass:NSString.class]) {
			[reading.text appendString:chunk[@"text"]];
		} else if ([type isEqualToString:@"thinking"]) {
			for (NSDictionary *part in [chunk[@"thinking"] isKindOfClass:NSArray.class] ? chunk[@"thinking"] : @[]) {
				if ([part isKindOfClass:NSDictionary.class] && [part[@"text"] isKindOfClass:NSString.class]) {
					[reading.reasoning appendString:part[@"text"]];
				}
			}
		} else if ([type isEqualToString:@"tool_reference"]) {
			NSMutableDictionary *citation = [NSMutableDictionary dictionaryWithObject:chunk forKey:@"raw"];
			citation[@"text"] = chunk[@"title"] ?: chunk[@"url"] ?: @"";
			citation[@"title"] = chunk[@"title"];
			citation[@"url"] = chunk[@"url"];
			[reading.citations addObject:citation];
		} else if ([type isEqualToString:@"tool_file"]) {
			NSMutableDictionary *file = [NSMutableDictionary dictionaryWithObject:chunk forKey:@"raw"];
			file[@"name"] = chunk[@"tool"] ?: type;
			file[@"output"] = chunk[@"file_id"];
			[reading.serverTools addObject:file];
			if ([self chunkNamesAnImage:chunk] && NFKConversationsString(chunk[@"file_id"]) != nil) {
				[reading.imageFiles addObject:chunk[@"file_id"]];
			}
		}
	}
}

- (BOOL)chunkNamesAnImage:(NSDictionary *)chunk
{
	NSString *fileType = [NFKConversationsString(chunk[@"file_type"]) lowercaseString] ?: @"";
	return [chunk[@"tool"] isEqual:@"image_generation"] || [@[ @"png", @"jpg", @"jpeg", @"webp", @"image/png", @"image/jpeg", @"image/webp" ] containsObject:fileType];
}

- (nullable id)parsedArguments:(id)arguments
{
	if ([arguments isKindOfClass:NSDictionary.class]) {
		return arguments;
	}
	if (![arguments isKindOfClass:NSString.class]) {
		return nil;
	}
	return NFKConversationsDictionary([NSJSONSerialization JSONObjectWithData:[arguments dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL]);
}

- (NSDictionary *)toolCallWithIdentifier:(id)identifier name:(id)name arguments:(id)arguments
{
	NSString *json = [arguments isKindOfClass:NSString.class] ? arguments : nil;
	if (json == nil && [arguments isKindOfClass:NSDictionary.class]) {
		NSData *data = [NSJSONSerialization dataWithJSONObject:arguments options:0 error:NULL];
		json = data != nil ? [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] : nil;
	}
	return @{ @"id": NFKConversationsString(identifier) ?: @"", @"name": NFKConversationsString(name) ?: @"",
			  @"arguments": [self parsedArguments:arguments] ?: @{}, @"argumentsJSON": json ?: @"{}" };
}

- (NFKInferenceResult *)resultFromReading:(NFKConversationsReading *)reading
								  request:(NFKInferenceRequest *)request
								  entries:(NSArray *)entries
									  raw:(NSDictionary *)raw
{
	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionaryWithObject:raw forKey:NFKRemoteBackendRawKey];
	NSString *text = [reading.text copy];
	if (text.length > 0) {
		outputs[NFKOutputText] = text;
	}
	if ([request.parameters[NFKParameterJSONSchema] isKindOfClass:NSDictionary.class] && text.length > 0) {
		id structured = [NSJSONSerialization JSONObjectWithData:[text dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
		if ([structured isKindOfClass:NSDictionary.class]) {
			outputs[NFKOutputStructured] = structured;
		}
	}
	if (reading.reasoning.length > 0) {
		outputs[NFKOutputReasoning] = [reading.reasoning copy];
	}
	if (reading.toolCalls.count > 0) {
		outputs[NFKOutputToolCalls] = [reading.toolCalls copy];
	}
	outputs[NFKOutputAssistantMessage] = NFKAssistantMessage(text, reading.toolCalls, NFKProviderTurnFormatMistralConversations,
															 self.endpointURL.absoluteString, entries.count > 0 ? entries : nil);
	if (reading.citations.count > 0) {
		outputs[NFKOutputCitations] = [reading.citations copy];
	}
	if (reading.serverTools.count > 0) {
		outputs[NFKOutputServerToolResults] = [reading.serverTools copy];
	}
	NSArray *images = self.downloadsImages ? [self imagesOfFiles:reading.imageFiles] : @[];
	if (images.count > 0) {
		outputs[NFKOutputImage] = images.firstObject;
	}
	if (images.count > 1) {
		outputs[NFKOutputImages] = images;
	}
	if (reading.conversationIdentifier != nil) {
		outputs[NFKOutputResponseIdentifier] = reading.conversationIdentifier;
	}
	NSDictionary *counts = NFKRemoteUsage(reading.usage[@"prompt_tokens"], nil, reading.usage[@"completion_tokens"], nil);
	if (counts != nil) {
		outputs[NFKOutputUsage] = counts;
	}
	return NFKResultWithFinishReason([NFKInferenceResult resultWithOutputs:outputs], reading.toolCalls.count > 0 ? @"tool_calls" : @"stop");
}

- (NSArray *)imagesOfFiles:(NSArray<NSString *> *)identifiers
{
	NSMutableArray *images = [NSMutableArray array];
	for (NSString *identifier in identifiers) {
		NSData *bytes = [self contentsOfFileWithIdentifier:identifier error:NULL];
		CVPixelBufferRef pixelBuffer = bytes != nil ? [NFKImageCoding pixelBufferWithImageData:bytes] : NULL;
		if (pixelBuffer != NULL) {
			[images addObject:(__bridge id)pixelBuffer];
			CVPixelBufferRelease(pixelBuffer);
		}
	}
	return images;
}

#pragma mark Stream

/*! Reads one event into the reading and answers whether the partial result changed. A function call
	streams its arguments in pieces under its tool_call_id. */
- (BOOL)readEvent:(NSDictionary *)event into:(NFKConversationsReading *)reading
{
	NSString *type = NFKConversationsString(event[@"type"]) ?: @"";
	if ([type isEqualToString:@"message.output.delta"]) {
		NSUInteger before = reading.text.length + reading.reasoning.length;
		[self readContent:event[@"content"] into:reading];
		return reading.text.length + reading.reasoning.length != before;
	}
	if ([type isEqualToString:@"function.call.delta"]) {
		NSString *identifier = NFKConversationsString(event[@"tool_call_id"]) ?: NFKConversationsString(event[@"id"]) ?: @"";
		NSMutableDictionary *call = reading.streamedCalls[identifier];
		if (call == nil) {
			call = [NSMutableDictionary dictionaryWithObjectsAndKeys:identifier, @"id", [NSMutableString string], @"arguments", nil];
			reading.streamedCalls[identifier] = call;
			[reading.streamedCallOrder addObject:identifier];
		}
		call[@"name"] = NFKConversationsString(event[@"name"]) ?: call[@"name"];
		if ([event[@"arguments"] isKindOfClass:NSString.class]) {
			[call[@"arguments"] appendString:event[@"arguments"]];
		}
		return NO;
	}
	if ([type isEqualToString:@"tool.execution.done"]) {
		[reading.serverTools addObject:@{ @"name": event[@"name"] ?: type, @"output": event[@"info"] ?: @{}, @"raw": event }];
	} else if ([type isEqualToString:@"conversation.response.started"]) {
		reading.conversationIdentifier = NFKConversationsString(event[@"conversation_id"]) ?: reading.conversationIdentifier;
	} else if ([type isEqualToString:@"conversation.response.done"]) {
		reading.usage = NFKConversationsDictionary(event[@"usage"]);
	} else if ([type isEqualToString:@"conversation.response.error"]) {
		reading.failure = NFKConversationsString(event[@"message"]) ?: @"the stream reported an error";
	}
	return NO;
}

- (void)collectStreamedCallsOf:(NFKConversationsReading *)reading
{
	for (NSString *identifier in reading.streamedCallOrder) {
		NSDictionary *call = reading.streamedCalls[identifier];
		[reading.toolCalls addObject:[self toolCallWithIdentifier:identifier name:call[@"name"] arguments:[call[@"arguments"] copy]]];
	}
}

#pragma mark Transport

- (NSMutableURLRequest *)requestTo:(NSURL *)url method:(NSString *)method
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.HTTPMethod = method;
	request.timeoutInterval = self.timeout;
	[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
	return request;
}

- (nullable NSData *)dataForRequest:(NSURLRequest *)request error:(NSError * _Nullable *)outError
{
	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:request response:&response error:&sendError];
	NSError *failure = data == nil ? sendError : [NFKRemoteTransport errorForResponse:response data:data];
	if (failure != nil || data == nil) {
		if (outError != NULL) {
			*outError = failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the conversation call failed"];
		}
		return nil;
	}
	return data;
}

- (nullable NSDictionary *)JSONForRequest:(NSURLRequest *)request error:(NSError * _Nullable *)outError
{
	NSData *data = [self dataForRequest:request error:outError];
	if (data == nil) {
		return nil;
	}
	NSDictionary *reply = NFKConversationsDictionary([NSJSONSerialization JSONObjectWithData:data options:0 error:NULL]);
	return reply ?: [self fail:outError code:kNFKError_InferenceBackendFailure reason:@"the reply is not a JSON object"];
}

- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	return [NFKRemoteTransport sendRequest:request session:self.session response:outResponse error:outError];
}

- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *line))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse * _Nullable, NSData * _Nullable, NSError * _Nullable))completionHandler
{
	return [NFKRemoteTransport streamRequest:request session:self.session lineHandler:lineHandler completionHandler:completionHandler];
}

- (nullable id)fail:(NSError * _Nullable *)outError code:(NFKInferenceError)code reason:(NSString *)reason
{
	if (outError != NULL) {
		*outError = [NFKRemoteTransport errorWithCode:code reason:reason];
	}
	return nil;
}

@end

#pragma mark - Batches

@implementation NFKMistralConversationsBackend (NFKRemoteBatch)

- (nullable NSString *)batchModelName
{
	return self.modelName;
}

- (nullable NSURLRequest *)batchURLRequestForRequest:(NFKInferenceRequest *)request error:(NSError * _Nullable *)outError
{
	return [self urlRequestForRequest:request streaming:NO error:outError];
}

- (nullable NFKInferenceResult *)batchResultForReply:(NSDictionary *)reply
											 request:(NFKInferenceRequest *)request
											   error:(NSError * _Nullable *)outError
{
	return [self resultFromReply:reply request:request error:outError];
}

@end
