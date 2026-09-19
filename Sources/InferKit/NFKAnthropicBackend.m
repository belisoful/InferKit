//
//  NFKAnthropicBackend.m
//  InferKit
//

#import <InferKit/NFKAnthropicBackend.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKRemoteTransport.h>
#import "NFKRemoteMediaSupport.h"
#import "NFK_ARC.h"

/*! The tool a schema is asked for through: the API has no response format, so the reply is
	forced into a tool whose input schema is the caller's. */
static NSString * const NFKAnthropicStructuredToolName = @"structured_output";

/*! The thinking budget each reasoning effort the contract names asks for. The Messages API takes a
	token budget rather than a named level, so the three levels are a table of budgets. The API's own
	floor is 1024 tokens, and max_tokens is raised where a budget would not fit under it. */
static NSDictionary<NSString *, NSNumber *> *NFKAnthropicThinkingBudgets(void)
{
	static NSDictionary<NSString *, NSNumber *> *budgets;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		budgets = @{ NFKReasoningEffortLight: @2048,
					 NFKReasoningEffortModerate: @8192,
					 NFKReasoningEffortDeep: @16384 };
	});
	return budgets;
}

/*! What a streamed reply assembles into: content blocks by index, each text, thinking, or a tool
	use, and the token counts the message events report. */
@interface NFKAnthropicStreamState : NSObject
@property (nonatomic, strong) NSMutableDictionary<NSNumber *, NSMutableDictionary *> *blocksByIndex;
@property (nonatomic, copy, nullable) NSDictionary<NSString *, NSNumber *> *usage;
@property (nonatomic, assign) BOOL finished;
@end

@implementation NFKAnthropicStreamState
- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_blocksByIndex = [NSMutableDictionary dictionary];
	}
	return self;
}
@end

@implementation NFKAnthropicBackend

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKAnthropicBackend *backend = [[self alloc] init];
	if (endpointURL != nil) {
		backend.endpointURL = endpointURL;
	}
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self) {
		_endpointURL = [NSURL URLWithString:@"https://api.anthropic.com/v1/messages"];
		_apiVersion = NFKAnthropicAPIVersion;
		_maxTokens = 1024;
		_timeout = 120;
		_session = NSURLSession.sharedSession;
	}
	return self;
}

- (NSString *)backendIdentifier
{
	return @"anthropic-messages";
}

- (NSSet<NSString *> *)supportedParameterKeys
{
	// The Messages API has no repetition penalty, so that core key has nothing to become here.
	return [NSSet setWithArray:@[ NFKParameterTools, NFKParameterJSONSchema, NFKParameterTemperature,
								  NFKParameterMaxTokens, NFKParameterTopP, NFKParameterTopK,
								  NFKParameterStopSequences, NFKParameterAudioOutput,
								  NFKParameterReasoningEffort ]];
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithArray:@[ NFKInputPrompt, NFKInputMessages, NFKInputImage, NFKInputImages,
								  NFKInputVideo, NFKInputAudio, NFKInputDocument, NFKInputDocuments ]];
}

- (BOOL)isReady
{
	return self.endpointURL != nil && self.modelName.length > 0;
}

#pragma mark Inference

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
		if (outError != NULL) { *outError = sendError; }
		return nil;
	}
	NSError *statusError = [NFKRemoteTransport errorForResponse:response data:responseData];
	if (statusError != nil) {
		if (outError != NULL) { *outError = statusError; }
		return nil;
	}
	NSDictionary *responseBody = [NSJSONSerialization JSONObjectWithData:responseData options:0 error:NULL];
	if (![responseBody isKindOfClass:NSDictionary.class]) {
		return [self failWithCode:kNFKError_InferenceBackendFailure
					  description:@"the endpoint returned a body that is not a JSON object"
							error:outError];
	}
	NSArray *content = [responseBody[@"content"] isKindOfClass:NSArray.class] ? responseBody[@"content"] : @[];
	return [self resultForContentBlocks:content
								  usage:[self usageInMessageBody:responseBody]
									raw:responseBody
					  expectsStructured:[self requestExpectsStructuredReply:request]];
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

	NFKAnthropicStreamState *state = [[NFKAnthropicStreamState alloc] init];
	BOOL expectsStructured = [self requestExpectsStructuredReply:request];
	void (^cancel)(void) = [self streamRequest:urlRequest lineHandler:^(NSString *line) {
		if (state.finished) {
			return;
		}
		NSString *payload = [NFKRemoteTransport SSEDataForLine:line];
		if (payload == nil) {
			return;
		}
		id event = [NSJSONSerialization JSONObjectWithData:[payload dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
		if (![event isKindOfClass:NSDictionary.class]) {
			return;
		}
		NSString *type = [event[@"type"] isKindOfClass:NSString.class] ? event[@"type"] : @"";
		if ([type isEqualToString:@"message_stop"]) {
			state.finished = YES;
			[job finishWithResult:[self resultFromStreamState:state expectsStructured:expectsStructured]];
			return;
		}
		if ([type isEqualToString:@"error"]) {
			state.finished = YES;
			NSDictionary *detail = [event[@"error"] isKindOfClass:NSDictionary.class] ? event[@"error"] : @{};
			NSString *message = [detail[@"message"] isKindOfClass:NSString.class] ? detail[@"message"] : @"the stream reported an error";
			[job finishWithError:[NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:message]];
			return;
		}
		if ([self applyStreamEvent:event type:type toState:state]) {
			NSMutableDictionary<NSString *, id> *partial = [NSMutableDictionary dictionary];
			partial[NFKRemoteBackendTextKey] = [self textInState:state];
			NSString *reasoning = [self reasoningInState:state];
			if (reasoning.length > 0) {
				partial[NFKOutputReasoning] = reasoning;
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
		[job finishWithResult:[self resultFromStreamState:state expectsStructured:expectsStructured]];
	}];
	job.cancellationHandler = cancel;
	return job;
}

#pragma mark Request

- (nullable NSMutableURLRequest *)urlRequestForRequest:(NFKInferenceRequest *)request
											 streaming:(BOOL)streaming
												 error:(NSError * _Nullable *)outError
{
	if (!self.isReady) {
		[self failWithCode:kNFKError_InferenceNotReady
			   description:@"the Anthropic backend needs an endpoint and a model name" error:outError];
		return nil;
	}
	// The Messages API takes images and documents and answers in text; audio in either direction
	// is refused here rather than silently dropped.
	if ([request inputForKey:NFKInputAudio] != nil) {
		[self failWithCode:kNFKError_InferenceUnsupported
			   description:@"the Messages API takes no audio input; transcribe it first" error:outError];
		return nil;
	}
	if ([request.parameters[NFKParameterAudioOutput] isKindOfClass:NSDictionary.class]) {
		[self failWithCode:kNFKError_InferenceUnsupported
			   description:@"the Messages API answers in text only; speak the reply through NFKRemoteSpeechBackend" error:outError];
		return nil;
	}
	id effort = [request parameterForKey:NFKParameterReasoningEffort];
	if (effort != nil && [self thinkingBudgetForEffort:effort] == nil) {
		[self failWithCode:kNFKError_InferenceUnsupported
			   description:@"NFKParameterReasoningEffort is light, moderate, deep, or a thinking budget in tokens" error:outError];
		return nil;
	}
	NFKRemoteAttachments *attachments = [NFKRemoteAttachments attachmentsForRequest:request error:outError];
	if (attachments == nil) {
		return nil;
	}
	NSMutableDictionary *body = [self bodyForRequest:request attachments:attachments];
	if (body == nil) {
		[self failWithCode:kNFKError_InferenceMissingInput
			   description:@"the request carries neither a prompt nor messages" error:outError];
		return nil;
	}
	if (streaming) {
		body[@"stream"] = @YES;
	}
	NSError *encodeError = nil;
	NSData *payload = [NSJSONSerialization dataWithJSONObject:body options:0 error:&encodeError];
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
	[NFKRemoteTransport authorizeRequest:urlRequest apiKey:self.apiKey style:NFKRemoteAPIStyleAnthropicMessages];
	// The transport sets the version this release was written against; apiVersion overrides it.
	[urlRequest setValue:self.apiVersion forHTTPHeaderField:@"anthropic-version"];
	return urlRequest;
}

/*! Builds the Messages request body: a system message lifted into the top-level field, media
	attached to the last user turn, tools in the API's shape, and a schema as a forced tool. */
- (nullable NSMutableDictionary *)bodyForRequest:(NFKInferenceRequest *)request attachments:(NFKRemoteAttachments *)attachments
{
	NSMutableDictionary *body = [NSMutableDictionary dictionary];
	body[@"model"] = self.modelName;

	NSNumber *maxTokens = [request parameterForKey:NFKParameterMaxTokens];
	body[@"max_tokens"] = [maxTokens isKindOfClass:NSNumber.class] ? maxTokens : @(self.maxTokens);

	// Extended thinking rules the sampling: the API refuses a request that sets temperature, top_p,
	// or top_k beside it, so those three are dropped where a reasoning effort is asked for. The
	// budget must also leave room under max_tokens for the answer itself.
	NSNumber *budget = [self thinkingBudgetForEffort:[request parameterForKey:NFKParameterReasoningEffort]];
	if (budget != nil) {
		body[@"thinking"] = @{ @"type": @"enabled", @"budget_tokens": budget };
		if ([body[@"max_tokens"] integerValue] <= budget.integerValue) {
			body[@"max_tokens"] = @(budget.integerValue + self.maxTokens);
		}
	}

	NSNumber *temperature = [request parameterForKey:NFKParameterTemperature];
	if ([temperature isKindOfClass:NSNumber.class] && budget == nil) {
		body[@"temperature"] = temperature;
	}

	// The Messages API names these three itself, so the core keys are renamed rather than dropped.
	NSNumber *topP = [request parameterForKey:NFKParameterTopP];
	if ([topP isKindOfClass:NSNumber.class] && budget == nil) {
		body[@"top_p"] = topP;
	}
	NSNumber *topK = [request parameterForKey:NFKParameterTopK];
	if ([topK isKindOfClass:NSNumber.class] && budget == nil) {
		body[@"top_k"] = topK;
	}
	NSArray<NSString *> *stopSequences = [request parameterForKey:NFKParameterStopSequences];
	if ([stopSequences isKindOfClass:NSArray.class]) {
		body[@"stop_sequences"] = stopSequences;
	}

	NSMutableArray *conversation = [NSMutableArray array];
	NSArray *messages = request.messages;
	if (messages != nil) {
		for (NSDictionary *message in messages) {
			// A system turn is a top-level field here, not a message with a role.
			if ([message[@"role"] isEqualToString:@"system"]) {
				body[@"system"] = message[@"content"];
				continue;
			}
			[conversation addObject:message];
		}
	} else if (request.prompt.length > 0) {
		[conversation addObject:@{ @"role": @"user", @"content": request.prompt }];
	}
	if (conversation.count == 0) {
		return nil;
	}
	if (!attachments.isEmpty && ![self attach:attachments toLastUserTurnIn:conversation]) {
		return nil;
	}
	body[@"messages"] = conversation;

	NSMutableArray *tools = [NSMutableArray array];
	NSArray *declared = request.parameters[NFKParameterTools];
	if ([declared isKindOfClass:NSArray.class]) {
		for (NSDictionary *tool in declared) {
			if ([tool isKindOfClass:NSDictionary.class]) {
				[tools addObject:[self wireToolForTool:tool]];
			}
		}
	}
	NSDictionary *schema = request.parameters[NFKParameterJSONSchema];
	if ([schema isKindOfClass:NSDictionary.class]) {
		[tools addObject:@{ @"name": NFKAnthropicStructuredToolName,
							@"description": @"Record the structured reply.",
							@"input_schema": schema }];
		body[@"tool_choice"] = @{ @"type": @"tool", @"name": NFKAnthropicStructuredToolName };
	}
	if (tools.count > 0) {
		body[@"tools"] = tools;
	}
	return body;
}

// The three levels the contract names come from the table; a numeric string is a budget in tokens,
// which is how a caller asks for one the table does not hold. Anything else is nil, and the request
// is refused rather than answered without the thinking it asked for.
- (nullable NSNumber *)thinkingBudgetForEffort:(nullable id)effort
{
	if (![effort isKindOfClass:NSString.class]) {
		return nil;
	}
	NSNumber *named = NFKAnthropicThinkingBudgets()[effort];
	if (named != nil) {
		return named;
	}
	NSScanner *scanner = [NSScanner scannerWithString:effort];
	NSInteger tokens = 0;
	if ([scanner scanInteger:&tokens] && scanner.isAtEnd && tokens > 0) {
		return @(tokens);
	}
	return nil;
}

// The contract's {name, description, parameters} is the API's {name, description, input_schema};
// an entry already carrying input_schema is in the wire shape and passes through.
- (NSDictionary *)wireToolForTool:(NSDictionary *)tool
{
	if (tool[@"input_schema"] != nil) {
		return tool;
	}
	NSMutableDictionary *wire = [NSMutableDictionary dictionary];
	wire[@"name"] = tool[@"name"] ?: @"";
	if (tool[@"description"] != nil) {
		wire[@"description"] = tool[@"description"];
	}
	wire[@"input_schema"] = tool[@"parameters"] ?: @{ @"type": @"object", @"properties": @{} };
	return wire;
}

// The media goes before the text, as the API's own examples order the blocks: images and sampled
// frames as base64 image blocks, documents as base64 PDF document blocks.
- (BOOL)attach:(NFKRemoteAttachments *)attachments toLastUserTurnIn:(NSMutableArray *)conversation
{
	for (NSInteger index = (NSInteger)conversation.count - 1; index >= 0; index--) {
		NSDictionary *message = conversation[index];
		if (![message isKindOfClass:NSDictionary.class] || ![message[@"role"] isEqual:@"user"]) {
			continue;
		}
		NSMutableArray *blocks = [NSMutableArray array];
		for (NSData *png in attachments.imagePNGs) {
			[blocks addObject:@{ @"type": @"image",
								 @"source": @{ @"type": @"base64", @"media_type": @"image/png",
											   @"data": [png base64EncodedStringWithOptions:0] } }];
		}
		for (NSDictionary *document in attachments.documents) {
			[blocks addObject:@{ @"type": @"document",
								 @"source": @{ @"type": @"base64", @"media_type": @"application/pdf",
											   @"data": [document[@"data"] base64EncodedStringWithOptions:0] },
								 @"title": document[@"filename"] }];
		}
		if ([message[@"content"] isKindOfClass:NSString.class]) {
			[blocks addObject:@{ @"type": @"text", @"text": message[@"content"] }];
		} else if ([message[@"content"] isKindOfClass:NSArray.class]) {
			[blocks addObjectsFromArray:message[@"content"]];
		}
		NSMutableDictionary *attached = [message mutableCopy];
		attached[@"content"] = blocks;
		conversation[index] = attached;
		return YES;
	}
	return NO;
}

- (BOOL)requestExpectsStructuredReply:(NFKInferenceRequest *)request
{
	return [request.parameters[NFKParameterJSONSchema] isKindOfClass:NSDictionary.class];
}

#pragma mark Response

// The counts ride under usage, with the input side split between what was read and what the cache
// served. The API reports no reasoning count: the thinking tokens are part of the output total.
- (nullable NSDictionary<NSString *, NSNumber *> *)usageInMessageBody:(nullable NSDictionary *)body
{
	NSDictionary *usage = [body[@"usage"] isKindOfClass:NSDictionary.class] ? body[@"usage"] : nil;
	if (usage == nil) {
		return nil;
	}
	return NFKRemoteUsage(usage[@"input_tokens"], usage[@"cache_read_input_tokens"],
						  usage[@"output_tokens"], nil);
}

/*! Joins the text blocks into the text output and reads the tool uses. The API returns a list of
	typed blocks, not one string; the forced structured tool's input is the structured output. */
- (NFKInferenceResult *)resultForContentBlocks:(NSArray *)content
											 usage:(nullable NSDictionary<NSString *, NSNumber *> *)usage
										   raw:(id)raw
							 expectsStructured:(BOOL)expectsStructured
{
	NSMutableDictionary *outputs = [NSMutableDictionary dictionary];
	NSMutableArray<NSString *> *pieces = [NSMutableArray array];
	NSMutableArray<NSString *> *reasoning = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *toolCalls = [NSMutableArray array];
	for (NSDictionary *block in content) {
		if (![block isKindOfClass:NSDictionary.class]) {
			continue;
		}
		if ([block[@"type"] isEqualToString:@"text"] && [block[@"text"] isKindOfClass:NSString.class]) {
			[pieces addObject:block[@"text"]];
			continue;
		}
		// A thinking block carries the chain; a redacted one carries encrypted bytes with nothing
		// to read, so it contributes no reasoning text.
		if ([block[@"type"] isEqualToString:@"thinking"] && [block[@"thinking"] isKindOfClass:NSString.class]) {
			[reasoning addObject:block[@"thinking"]];
			continue;
		}
		if (![block[@"type"] isEqualToString:@"tool_use"] || ![block[@"name"] isKindOfClass:NSString.class]) {
			continue;
		}
		NSDictionary *input = [block[@"input"] isKindOfClass:NSDictionary.class] ? block[@"input"] : @{};
		if (expectsStructured && [block[@"name"] isEqualToString:NFKAnthropicStructuredToolName]) {
			outputs[NFKOutputStructured] = input;
			continue;
		}
		NSData *json = [NSJSONSerialization dataWithJSONObject:input options:0 error:NULL];
		[toolCalls addObject:@{ @"id": [block[@"id"] isKindOfClass:NSString.class] ? block[@"id"] : @"",
								@"name": block[@"name"],
								@"arguments": input,
								@"argumentsJSON": json != nil ? [[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding] : @"{}" }];
	}
	if (pieces.count > 0) {
		outputs[NFKRemoteBackendTextKey] = [pieces componentsJoinedByString:@""];
	}
	if (reasoning.count > 0) {
		outputs[NFKOutputReasoning] = [reasoning componentsJoinedByString:@"\n"];
	}
	if (toolCalls.count > 0) {
		outputs[NFKOutputToolCalls] = toolCalls;
	}
	if (usage != nil) {
		outputs[NFKOutputUsage] = usage;
	}
	outputs[NFKRemoteBackendRawKey] = raw;
	return [NFKInferenceResult resultWithOutputs:outputs];
}

#pragma mark Streaming

// content_block_start opens a block; content_block_delta appends text or partial JSON to it.
// Returns whether the event carried text, which is what a partial result is worth reporting for.
- (BOOL)applyStreamEvent:(NSDictionary *)event type:(NSString *)type toState:(NFKAnthropicStreamState *)state
{
	// message_start opens with what the request cost; message_delta closes with what the reply cost,
	// so the two together are the turn's counts.
	if ([type isEqualToString:@"message_start"] || [type isEqualToString:@"message_delta"]) {
		NSDictionary *carrier = [event[@"message"] isKindOfClass:NSDictionary.class] ? event[@"message"] : event;
		NSDictionary<NSString *, NSNumber *> *reported = [self usageInMessageBody:carrier];
		if (reported != nil) {
			NSMutableDictionary<NSString *, NSNumber *> *usage = [state.usage mutableCopy] ?: [NSMutableDictionary dictionary];
			[usage addEntriesFromDictionary:reported];
			state.usage = usage;
		}
	}
	NSNumber *index = [event[@"index"] isKindOfClass:NSNumber.class] ? event[@"index"] : nil;
	if ([type isEqualToString:@"content_block_start"] && index != nil) {
		NSDictionary *block = [event[@"content_block"] isKindOfClass:NSDictionary.class] ? event[@"content_block"] : @{};
		NSMutableDictionary *opened = [block mutableCopy];
		if ([opened[@"type"] isEqual:@"text"]) {
			opened[@"text"] = [opened[@"text"] isKindOfClass:NSString.class] ? opened[@"text"] : @"";
		} else if ([opened[@"type"] isEqual:@"thinking"]) {
			opened[@"thinking"] = [opened[@"thinking"] isKindOfClass:NSString.class] ? opened[@"thinking"] : @"";
		} else if ([opened[@"type"] isEqual:@"tool_use"]) {
			opened[@"partial_json"] = @"";
		}
		state.blocksByIndex[index] = opened;
		return NO;
	}
	if (![type isEqualToString:@"content_block_delta"] || index == nil) {
		return NO;
	}
	NSMutableDictionary *block = state.blocksByIndex[index];
	NSDictionary *delta = [event[@"delta"] isKindOfClass:NSDictionary.class] ? event[@"delta"] : nil;
	if (block == nil || delta == nil) {
		return NO;
	}
	if ([delta[@"type"] isEqual:@"text_delta"] && [delta[@"text"] isKindOfClass:NSString.class]) {
		block[@"text"] = [(block[@"text"] ?: @"") stringByAppendingString:delta[@"text"]];
		return [delta[@"text"] length] > 0;
	}
	if ([delta[@"type"] isEqual:@"thinking_delta"] && [delta[@"thinking"] isKindOfClass:NSString.class]) {
		block[@"thinking"] = [(block[@"thinking"] ?: @"") stringByAppendingString:delta[@"thinking"]];
		return [delta[@"thinking"] length] > 0;
	}
	if ([delta[@"type"] isEqual:@"input_json_delta"] && [delta[@"partial_json"] isKindOfClass:NSString.class]) {
		block[@"partial_json"] = [(block[@"partial_json"] ?: @"") stringByAppendingString:delta[@"partial_json"]];
	}
	return NO;
}

// The blocks of one type, in the order the stream opened them; a block's own text rides under a
// field named for the block, which is what the API's deltas append to.
- (NSArray<NSString *> *)piecesOfBlockType:(NSString *)type inState:(NFKAnthropicStreamState *)state
{
	NSMutableArray<NSString *> *pieces = [NSMutableArray array];
	for (NSNumber *index in [state.blocksByIndex.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		NSDictionary *block = state.blocksByIndex[index];
		if ([block[@"type"] isEqual:type] && [block[type] isKindOfClass:NSString.class]) {
			[pieces addObject:block[type]];
		}
	}
	return pieces;
}

- (NSString *)textInState:(NFKAnthropicStreamState *)state
{
	return [[self piecesOfBlockType:@"text" inState:state] componentsJoinedByString:@""];
}

- (NSString *)reasoningInState:(NFKAnthropicStreamState *)state
{
	return [[self piecesOfBlockType:@"thinking" inState:state] componentsJoinedByString:@"\n"];
}

// The assembled blocks are read the way a blocking reply's are; a tool use's input is the parse of
// its accumulated partial JSON (an empty input streams as no deltas at all).
- (NFKInferenceResult *)resultFromStreamState:(NFKAnthropicStreamState *)state expectsStructured:(BOOL)expectsStructured
{
	NSMutableArray *content = [NSMutableArray array];
	for (NSNumber *index in [state.blocksByIndex.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		NSMutableDictionary *block = [state.blocksByIndex[index] mutableCopy];
		if ([block[@"type"] isEqual:@"tool_use"]) {
			NSString *partial = block[@"partial_json"];
			id input = partial.length > 0
				? [NSJSONSerialization JSONObjectWithData:[partial dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL] : nil;
			block[@"input"] = [input isKindOfClass:NSDictionary.class] ? input : @{};
			[block removeObjectForKey:@"partial_json"];
		}
		[content addObject:block];
	}
	return [self resultForContentBlocks:content
								  usage:state.usage
									raw:@{ @"role": @"assistant", @"content": content }
					  expectsStructured:expectsStructured];
}

#pragma mark Transport

/*! Overridden in tests to stage a response without a network. */
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

- (nullable NFKInferenceResult *)failWithCode:(NSInteger)code
								  description:(NSString *)description
										error:(NSError * _Nullable *)outError
{
	if (outError != NULL) {
		*outError = [NSError errorWithDomain:NFKInferenceErrorDomain
										code:code
									userInfo:@{ NSLocalizedDescriptionKey: description }];
	}
	return nil;
}

@end
