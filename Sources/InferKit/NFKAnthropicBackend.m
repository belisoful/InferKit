//
//  NFKAnthropicBackend.m
//  InferKit
//

#import <InferKit/NFKAnthropicBackend.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import "NFKRemoteTransportPrivate.h"
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKRemoteTransport.h>
#import "NFKRemoteMediaSupport.h"
#import "NFKToolMessageSupport.h"
#import <InferKit/NFKToolMessages.h>
#import <InferKit/NFKRemoteFileStore.h>
#import <InferKit/NFKRemoteProvider.h>
#import "NFK_ARC.h"
#import "NFKAnthropicBackendPrivate.h"
#import "NFKRemoteBatchPrivate.h"
#import "NFKJobGuard.h"

/*! The tool a schema is asked for through: the API has no response format, so the reply is
	forced into a tool whose input schema is the caller's. */
/*! The code-execution tool a skill runs in, Anthropic's current version (2026-10-23). A request that
	names its own code-execution tool keeps it. */
static NSString * const NFKAnthropicCodeExecutionToolType = @"code_execution_20260521";
static NSString * const NFKAnthropicStructuredToolName = @"structured_output";

/*! The beta Claude's MCP connector is served under. */
static NSString * const NFKAnthropicMCPConnectorBeta = @"mcp-client-2025-11-20";

NSString * const NFKAnthropicBackendInputTransformationsKey = @"inputTransformations";

/*! The beta that adds thinking.block_binding.prefix_mismatch_behavior to a request and the
	input_transformations array to its reply. */
static NSString * const NFKAnthropicThinkingBindingBeta = @"thinking-binding-controls-2026-08-01";

/*! The sentence that marks the 400 a failed prefix check returns. Anthropic's page on the error says to
	match on it alone, since the rest of the message can change. */
static NSString * const NFKAnthropicPrefixMismatchSentence = @"The block is bound to a different conversation";

/*! The families that run the prefix check on replayed thinking blocks, which are the models
	Anthropic's prefix-check page lists (2026-11-18). All of them think adaptively when a request
	names no thinking. */
static NSArray<NSString *> *NFKAnthropicPrefixCheckFamilies(void)
{
	return @[ @"claude-fable-5-2", @"claude-mythos-5-2", @"claude-fable-5-1", @"claude-mythos-5-1",
			  @"claude-opus-5-5", @"claude-sonnet-5-5", @"claude-haiku-5-5" ];
}

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

/*! The output_config effort each reasoning effort the contract names asks for, on a model that
	thinks adaptively. */
static NSDictionary<NSString *, NSString *> *NFKAnthropicEffortLevels(void)
{
	static NSDictionary<NSString *, NSString *> *levels;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		levels = @{ NFKReasoningEffortLight: @"low",
					NFKReasoningEffortModerate: @"medium",
					NFKReasoningEffortDeep: @"high" };
	});
	return levels;
}

/*! The families that take a thinking budget and sampling: every model released before Claude Opus
	4.6. A family is matched anywhere in the name, so a cloud spelling (anthropic.claude-…,
	claude-…@date) matches too. */
static NSArray<NSString *> *NFKAnthropicBudgetThinkingFamilies(void)
{
	return @[ @"claude-3", @"claude-opus-4-0", @"claude-sonnet-4-0", @"claude-opus-4-2025",
			  @"claude-sonnet-4-2025", @"claude-opus-4@", @"claude-sonnet-4@", @"claude-opus-4-1",
			  @"claude-opus-4-5", @"claude-sonnet-4-5", @"claude-haiku-4-5" ];
}

/*! The families that think adaptively and still take sampling, and whose thinking display already
	defaults to a summary: Claude Opus 4.6 and Claude Sonnet 4.6. */
static NSArray<NSString *> *NFKAnthropicSummarizingFamilies(void)
{
	return @[ @"claude-opus-4-6", @"claude-sonnet-4-6" ];
}

/*! The families that refuse a forced tool_choice, so a schema goes through output_config.format. */
static NSArray<NSString *> *NFKAnthropicUnforcedToolFamilies(void)
{
	return @[ @"claude-opus-5-5", @"claude-sonnet-5-5", @"claude-fable-5-1", @"claude-mythos-5-1",
			  @"claude-fable-5-2", @"claude-mythos-5-2" ];
}

static BOOL NFKAnthropicModelIsInFamilies(NSString * _Nullable model, NSArray<NSString *> *families)
{
	for (NSString *family in families) {
		if ([model containsString:family]) {
			return YES;
		}
	}
	return NO;
}

/*! Whether the model is one of Meta's Muse models, which Meta's Model API serves on a Messages route.
	That route takes temperature and top_p beside the reasoning and answers 400 to top_k, to
	stop_sequences, and to a named tool_choice. */
static BOOL NFKAnthropicModelIsMuse(NSString * _Nullable model)
{
	return [model hasPrefix:@"muse-"];
}

BOOL NFKAnthropicProviderTakesBearerToken(NFKRemoteProvider *provider)
{
	return [provider.identifier isEqualToString:@"meta"];
}

/*! What a streamed reply assembles into: content blocks by index, each text, thinking, or a tool
	use, and the token counts the message events report. */
@interface NFKAnthropicStreamState : NSObject
@property (nonatomic, strong) NSMutableDictionary<NSNumber *, NSMutableDictionary *> *blocksByIndex;
@property (nonatomic, copy, nullable) NSDictionary<NSString *, NSNumber *> *usage;
@property (nonatomic, copy, nullable) NSString *stopReason;
/*! The blocks the backend removed before this stream was sent. */
@property (nonatomic, copy, nullable) NSArray<NSDictionary *> *removedThinking;
/*! The input_transformations the reply reports under the thinking-binding beta. */
@property (nonatomic, copy, nullable) NSArray<NSDictionary *> *inputTransformations;
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

/*! The max_tokens sent where neither the request, the backend, nor the Models API names one. */
static const NSInteger NFKAnthropicFallbackMaxTokens = 1024;

/*! The longest a Models API lookup waits before a run sends the fallback. */
static const NSTimeInterval NFKAnthropicModelLookupTimeout = 10;

/*! The beta that has a model released before Claude 4.5 accept input plus max_tokens past its context
	window and stop with model_context_window_exceeded, as later models do unasked. */
static NSString * const NFKAnthropicContextWindowExceededBeta = @"model-context-window-exceeded-2025-08-26";

/*! The families released before Claude 4.5, which refuse input plus max_tokens past the context window
	unless the request carries NFKAnthropicContextWindowExceededBeta. */
static NSArray<NSString *> *NFKAnthropicContextErrorFamilies(void)
{
	return @[ @"claude-3", @"claude-opus-4-0", @"claude-sonnet-4-0", @"claude-opus-4-2025",
			  @"claude-sonnet-4-2025", @"claude-opus-4@", @"claude-sonnet-4@", @"claude-opus-4-1" ];
}

@interface NFKAnthropicBackend ()
/*! The max_tokens the Models API stated in a 200 answer, or NSNull where that answer stated none, by
	endpoint and model name. Read and written under @synchronized (self). */
@property (nonatomic, strong) NSMutableDictionary<NSString *, id> *modelMaximums;
/*! Runs one Models API lookup at a time, so concurrent first runs ask once. */
@property (nonatomic, strong) dispatch_queue_t lookupQueue;
/*! Whether the model takes a thinking budget in tokens rather than an adaptive effort. */
@property (nonatomic, readonly) BOOL usesThinkingBudget;
/*! Whether the model takes temperature, top_p, and top_k when it is not thinking. */
@property (nonatomic, readonly) BOOL acceptsSampling;
/*! Whether the model takes a forced tool_choice. */
@property (nonatomic, readonly) BOOL acceptsForcedTool;
/*! Whether the model is one of Meta's Muse models. */
@property (nonatomic, readonly) BOOL servesMuseModel;
/*! The prefix_mismatch_behavior value a request to this model carries, or nil where it carries none:
	the behavior is unset, or the model runs no prefix check. */
@property (nonatomic, readonly, nullable) NSString *prefixMismatchSetting;
- (nullable NSString *)effortLevelForEffort:(nullable id)effort;
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

+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
{
	NSArray<NSString *> *serving = @[ @"anthropic", @"opencode", @"opencode-go", @"meta", @"inferkit" ];
	if (![serving containsObject:provider.identifier]) {
		return nil;
	}
	NFKAnthropicBackend *backend = [self backendWithEndpointURL:[provider URLForPath:@"messages"]];
	backend.authorizesWithBearerToken = NFKAnthropicProviderTakesBearerToken(provider);
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
	if (self) {
		_endpointURL = [NSURL URLWithString:@"https://api.anthropic.com/v1/messages"];
		_apiVersion = NFKAnthropicAPIVersion;
		_timeout = 120;
		_recoversFromPrefixMismatch = YES;
		_modelMaximums = [NSMutableDictionary dictionary];
		_lookupQueue = dispatch_queue_create("com.inferkit.anthropic.model-lookup", DISPATCH_QUEUE_SERIAL);
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
								  NFKParameterReasoningEffort, NFKParameterMCPServers, NFKParameterSkills ]];
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithArray:@[ NFKInputPrompt, NFKInputMessages, NFKInputImage, NFKInputImages,
								  NFKInputVideo, NFKInputAudio, NFKInputDocument, NFKInputDocuments ]];
}

- (NSSet<NSString *> *)supportedOutputKeys
{
	return [NSSet setWithArray:@[ NFKRemoteBackendTextKey, NFKOutputStructured, NFKOutputReasoning,
								  NFKOutputToolCalls, NFKOutputAssistantMessage, NFKOutputUsage, NFKOutputFinishReason,
								  NFKOutputCitations, NFKOutputServerToolResults, NFKRemoteBackendRawKey,
								  NFKAnthropicBackendInputTransformationsKey ]];
}

- (BOOL)isReady
{
	return self.endpointURL != nil && self.modelName.length > 0;
}

#pragma mark Inference

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError * _Nullable *)outError
{
	BOOL modelMaximum = NO;
	NSInteger maxTokens = [self maxTokensForRequest:request lookingUp:self.isReady modelMaximum:&modelMaximum];
	if (modelMaximum && maxTokens > NFKAnthropicFallbackMaxTokens) {
		return [self resultByStreamingRequest:request maxTokens:maxTokens error:outError];
	}
	NSMutableURLRequest *urlRequest = [self urlRequestForRequest:request streaming:NO maxTokens:maxTokens
												   modelMaximum:modelMaximum error:outError];
	if (urlRequest == nil) {
		return nil;
	}
	NSArray<NSDictionary *> *removedThinking = nil;
	NSData *responseData = [self dataBySendingRequest:urlRequest removedThinking:&removedThinking error:outError];
	if (responseData == nil) {
		return nil;
	}
	NSDictionary *responseBody = [NSJSONSerialization JSONObjectWithData:responseData options:0 error:NULL];
	if (![responseBody isKindOfClass:NSDictionary.class]) {
		return [self failWithCode:kNFKError_InferenceBackendFailure
					  description:@"the endpoint returned a body that is not a JSON object"
							error:outError];
	}
	if ([responseBody[@"stop_reason"] isEqual:@"refusal"]) {
		return [self failWithCode:kNFKError_InferenceRefused description:@"the model declined to answer" error:outError];
	}
	NSArray *content = [responseBody[@"content"] isKindOfClass:NSArray.class] ? responseBody[@"content"] : @[];
	return [self resultForContentBlocks:content
								  usage:[self usageInMessageBody:responseBody]
									raw:responseBody
							 stopReason:responseBody[@"stop_reason"]
						transformations:[self transformationsAfter:removedThinking reportedIn:responseBody]
								request:request];
}

// A run that has to ask the Models API for the model's maximum asks off the caller's thread, so a slow
// network never holds the thread that submitted it.
- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	if (self.isReady && [self needsModelMaximumLookupForRequest:request]) {
		[job reportProgress:-1.0];
		dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			BOOL modelMaximum = NO;
			NSInteger maxTokens = [self maxTokensForRequest:request lookingUp:YES modelMaximum:&modelMaximum];
			if (job.status != NFKInferenceJobStatusCancelled) {
				[self streamRequest:request maxTokens:maxTokens modelMaximum:modelMaximum intoJob:job];
			}
		});
		return job;
	}
	BOOL modelMaximum = NO;
	NSInteger maxTokens = [self maxTokensForRequest:request lookingUp:NO modelMaximum:&modelMaximum];
	[self streamRequest:request maxTokens:maxTokens modelMaximum:modelMaximum intoJob:job];
	return job;
}

/*! Streams the request into job, sending maxTokens. */
- (void)streamRequest:(NFKInferenceRequest *)request maxTokens:(NSInteger)maxTokens modelMaximum:(BOOL)modelMaximum
			  intoJob:(NFKInferenceJob *)job
{
	NSError *error = nil;
	NSMutableURLRequest *urlRequest = [self urlRequestForRequest:request streaming:YES maxTokens:maxTokens
												   modelMaximum:modelMaximum error:&error];
	if (urlRequest == nil) {
		[job finishWithError:error];
		return;
	}
	[job reportProgress:-1.0];
	[self streamURLRequest:urlRequest forRequest:request removedThinking:nil intoJob:job];
}

/*! Streams urlRequest into job. A stream the prefix check refuses is sent once more without the
	mismatched thinking; removedThinking names the blocks an earlier attempt removed, and is nil for
	the first attempt. */
- (void)streamURLRequest:(NSMutableURLRequest *)urlRequest
			  forRequest:(NFKInferenceRequest *)request
		 removedThinking:(nullable NSArray<NSDictionary *> *)removedThinking
				 intoJob:(NFKInferenceJob *)job
{
	NFKAnthropicStreamState *state = [[NFKAnthropicStreamState alloc] init];
	state.removedThinking = removedThinking;
	NSArray<NSString *> *clientStops = [self clientStopsForRequest:request];
	__block BOOL resent = NO;
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
			[self finishJob:job fromStreamState:state request:request];
			return;
		}
		if ([type isEqualToString:@"error"]) {
			state.finished = YES;
			NSDictionary *detail = [event[@"error"] isKindOfClass:NSDictionary.class] ? event[@"error"] : @{};
			NSString *message = [detail[@"message"] isKindOfClass:NSString.class] ? detail[@"message"] : @"the stream reported an error";
			[job finishWithError:[NFKRemoteTransport errorWithCode:[self codeForStreamErrorType:detail[@"type"]] reason:message]];
			return;
		}
		if ([self applyStreamEvent:event type:type toState:state]) {
			NSMutableDictionary<NSString *, id> *partial = [NSMutableDictionary dictionary];
			NSString *text = [self textInState:state];
			partial[NFKRemoteBackendTextKey] = clientStops != nil ? NFKRemoteTextBeforeStops(text, clientStops) : text;
			NSString *reasoning = [self reasoningInState:state];
			if (reasoning.length > 0) {
				partial[NFKOutputReasoning] = reasoning;
			}
			[job reportProgress:-1.0 partialResult:[NFKInferenceResult resultWithOutputs:partial]];
		}
	} completionHandler:^(NSHTTPURLResponse * _Nullable response, NSData * _Nullable errorBody, NSError * _Nullable streamError) {
		NFKJobRunGuarded(job, ^{
			if (state.finished || job.status == NFKInferenceJobStatusCancelled) {
				return;
			}
			state.finished = YES;
			NSError *failure = streamError ?: [NFKRemoteTransport errorForResponse:response data:errorBody];
			NSArray<NSDictionary *> *removed = nil;
			NSMutableURLRequest *retry = removedThinking == nil
				? [self requestByRemovingMismatchedThinkingFrom:urlRequest afterError:failure removed:&removed] : nil;
			if (retry != nil) {
				resent = YES;
				[self streamURLRequest:retry forRequest:request removedThinking:removed intoJob:job];
				return;
			}
			if (failure != nil) {
				[job finishWithError:failure];
				return;
			}
			[self finishJob:job fromStreamState:state request:request];
		});
	}];
	// A stream that ended inside the call above and was sent again has set a handler of its own.
	if (resent) {
		return;
	}
	job.cancellationHandler = cancel;
	// A cancel that arrived while the lookup ran found no handler to call.
	if (job.status == NFKInferenceJobStatusCancelled) {
		cancel();
	}
}

/*! Runs the request as a stream sending maxTokens and waits for its end, as a run that sends the model's
	maximum does. */
- (nullable NFKInferenceResult *)resultByStreamingRequest:(NFKInferenceRequest *)request maxTokens:(NSInteger)maxTokens
													error:(NSError * _Nullable *)outError
{
	dispatch_semaphore_t finished = dispatch_semaphore_create(0);
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	[self streamRequest:request maxTokens:maxTokens modelMaximum:YES intoJob:job];
	job.completionHandler = ^(NFKInferenceJob *ended) {
		dispatch_semaphore_signal(finished);
	};
	dispatch_semaphore_wait(finished, DISPATCH_TIME_FOREVER);
	if (job.result == nil && outError != NULL) {
		*outError = job.error;
	}
	return job.result;
}

#pragma mark Output limit

/*! The max_tokens a run sends: the request's NFKParameterMaxTokens, else the backend's maxTokens, else the
	model's maximum (asked of the Models API when lookingUp, else as remembered), else 1024. outModelMaximum says
	whether it is the model's maximum. */
- (NSInteger)maxTokensForRequest:(NFKInferenceRequest *)request lookingUp:(BOOL)lookingUp
					modelMaximum:(nullable BOOL *)outModelMaximum
{
	NSNumber *requested = [request parameterForKey:NFKParameterMaxTokens];
	if ([requested isKindOfClass:NSNumber.class]) {
		return requested.integerValue;
	}
	if (self.maxTokens > 0) {
		return self.maxTokens;
	}
	NSNumber *maximum = lookingUp ? [self modelMaximumOutputTokens] : [self rememberedModelMaximum];
	if (maximum.integerValue > 0) {
		if (outModelMaximum != NULL) {
			*outModelMaximum = YES;
		}
		return maximum.integerValue;
	}
	return NFKAnthropicFallbackMaxTokens;
}

/*! Whether a run of the request asks the Models API for the model's maximum: it sets no limit, and no 200
	answer for the model is remembered. */
- (BOOL)needsModelMaximumLookupForRequest:(NFKInferenceRequest *)request
{
	if ([[request parameterForKey:NFKParameterMaxTokens] isKindOfClass:NSNumber.class] || self.maxTokens > 0) {
		return NO;
	}
	NSString *key = [self modelMaximumKey];
	@synchronized (self) {
		return key != nil && self.modelMaximums[key] == nil;
	}
}

/*! The key a model's remembered maximum is kept under: the endpoint and the model name, or nil without a
	key, a model, or an endpoint to ask. */
- (nullable NSString *)modelMaximumKey
{
	if (self.modelName.length == 0 || self.apiKey.length == 0 || self.endpointURL == nil) {
		return nil;
	}
	return [NSString stringWithFormat:@"%@ %@", self.endpointURL.absoluteString, self.modelName];
}

/*! The maximum a 200 answer stated for the model, without asking. */
- (nullable NSNumber *)rememberedModelMaximum
{
	NSString *key = [self modelMaximumKey];
	@synchronized (self) {
		id known = key != nil ? self.modelMaximums[key] : nil;
		return [known isKindOfClass:NSNumber.class] ? known : nil;
	}
}

/*! The max_tokens ceiling the Models API states for modelName (GET …/v1/models/{model}), asked with the
	backend's key and NFKAnthropicModelLookupTimeout. Lookups run one at a time, so a run that waited behind
	another reads its answer. Only a 200 answer is remembered: a failure, a 401, or an overload is asked again
	by a later run, which meanwhile sends 1024. */
- (nullable NSNumber *)modelMaximumOutputTokens
{
	NSString *key = [self modelMaximumKey];
	if (key == nil) {
		return nil;
	}
	__block NSNumber *maximum = nil;
	dispatch_sync(self.lookupQueue, ^{
		@synchronized (self) {
			id known = self.modelMaximums[key];
			if (known != nil) {
				maximum = [known isKindOfClass:NSNumber.class] ? known : nil;
				return;
			}
		}
		NSURL *url = [[self.endpointURL.URLByDeletingLastPathComponent URLByAppendingPathComponent:@"models"]
					  URLByAppendingPathComponent:self.modelName];
		NSMutableURLRequest *lookup = [NSMutableURLRequest requestWithURL:url];
		lookup.HTTPMethod = @"GET";
		lookup.timeoutInterval = MIN(self.timeout, NFKAnthropicModelLookupTimeout);
		[lookup setValue:@"application/json" forHTTPHeaderField:@"Accept"];
		[self authorizeURLRequest:lookup];
		NSHTTPURLResponse *response = nil;
		NSData *data = [self sendRequest:lookup response:&response error:NULL];
		if (data == nil || response.statusCode != 200) {
			return;
		}
		id entry = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
		maximum = [entry isKindOfClass:NSDictionary.class] && [entry[@"max_tokens"] isKindOfClass:NSNumber.class]
			&& [entry[@"max_tokens"] integerValue] > 0 ? entry[@"max_tokens"] : nil;
		@synchronized (self) {
			self.modelMaximums[key] = maximum ?: NSNull.null;
		}
	});
	return maximum;
}

/*! A stream that ended with a refusal fails the job as one; any other end delivers what streamed. */
- (void)finishJob:(NFKInferenceJob *)job fromStreamState:(NFKAnthropicStreamState *)state request:(NFKInferenceRequest *)request
{
	if ([state.stopReason isEqual:@"refusal"]) {
		[job finishWithError:[NFKRemoteTransport errorWithCode:kNFKError_InferenceRefused reason:@"the model declined to answer"]];
		return;
	}
	[job finishWithResult:[self resultFromStreamState:state request:request]];
}

/*! The core code a streamed error event's type maps to: an overload or a rate limit says back off, an
	invalid request says change it, and the rest is the backend's failure. */
- (NFKInferenceError)codeForStreamErrorType:(nullable id)type
{
	if ([type isEqual:@"overloaded_error"] || [type isEqual:@"rate_limit_error"]) {
		return kNFKError_InferenceRateLimited;
	}
	if ([type isEqual:@"invalid_request_error"]) {
		return kNFKError_InferenceRefused;
	}
	return kNFKError_InferenceBackendFailure;
}

#pragma mark Token counting

- (nullable NSDictionary<NSString *, id> *)countTokensBodyForRequest:(NFKInferenceRequest *)request
															   error:(NSError * _Nullable *)outError
{
	NSMutableURLRequest *urlRequest = [self urlRequestForRequest:request streaming:NO
													   maxTokens:[self maxTokensForRequest:request lookingUp:NO modelMaximum:NULL]
												   modelMaximum:NO error:outError];
	if (urlRequest == nil) {
		return nil;
	}
	NSDictionary *runBody = [NSJSONSerialization JSONObjectWithData:urlRequest.HTTPBody options:0 error:NULL];
	NSMutableDictionary *body = [NSMutableDictionary dictionary];
	for (NSString *key in @[ @"model", @"messages", @"system", @"tools", @"tool_choice", @"thinking", @"mcp_servers" ]) {
		body[key] = runBody[key];
	}
	return body;
}

- (nullable NSNumber *)tokenCountForRequest:(NFKInferenceRequest *)request error:(NSError * _Nullable *)outError
{
	NSMutableURLRequest *urlRequest = [self urlRequestForRequest:request streaming:NO
													   maxTokens:[self maxTokensForRequest:request lookingUp:NO modelMaximum:NULL]
												   modelMaximum:NO error:outError];
	NSDictionary *body = urlRequest != nil ? [self countTokensBodyForRequest:request error:outError] : nil;
	if (body == nil) {
		return nil;
	}
	urlRequest.URL = [self.endpointURL URLByAppendingPathComponent:@"count_tokens"];
	urlRequest.HTTPBody = [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL];
	NSData *data = [self dataBySendingRequest:urlRequest removedThinking:NULL error:outError];
	if (data == nil) {
		return nil;
	}
	NSDictionary *reply = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	NSNumber *count = [reply isKindOfClass:NSDictionary.class] && [reply[@"input_tokens"] isKindOfClass:NSNumber.class] ? reply[@"input_tokens"] : nil;
	if (count == nil) {
		[self failWithCode:kNFKError_InferenceBackendFailure description:@"the count reply carries no input_tokens" error:outError];
	}
	return count;
}

#pragma mark Request

- (nullable NSMutableURLRequest *)urlRequestForRequest:(NFKInferenceRequest *)request
											 streaming:(BOOL)streaming
											 maxTokens:(NSInteger)maxTokens
										  modelMaximum:(BOOL)modelMaximum
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
	if (effort != nil && self.usesThinkingBudget && [self thinkingBudgetForEffort:effort] == nil) {
		[self failWithCode:kNFKError_InferenceUnsupported
			   description:@"NFKParameterReasoningEffort is light, moderate, deep, or a thinking budget in tokens" error:outError];
		return nil;
	}
	if (effort != nil && !self.usesThinkingBudget && [self effortLevelForEffort:effort] == nil) {
		[self failWithCode:kNFKError_InferenceUnsupported
			   description:@"this model takes a named effort level, not a thinking budget in tokens" error:outError];
		return nil;
	}
	// Meta's route reads a whole clip, soundtrack included, as a video block; Anthropic's reads frames.
	NFKRemoteAttachments *attachments = [NFKRemoteAttachments attachmentsForRequest:request keepsVideo:self.servesMuseModel error:outError];
	if (attachments == nil) {
		return nil;
	}
	NSMutableDictionary *body = [self bodyForRequest:request attachments:attachments maxTokens:maxTokens];
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
	[self authorizeURLRequest:urlRequest];
	NSMutableArray<NSString *> *betas = [NSMutableArray array];
	if ([body[@"mcp_servers"] count] > 0) {
		[betas addObject:NFKAnthropicMCPConnectorBeta];
	}
	if (modelMaximum && NFKAnthropicModelIsInFamilies(self.modelName, NFKAnthropicContextErrorFamilies())) {
		[betas addObject:NFKAnthropicContextWindowExceededBeta];
	}
	if (body[@"thinking"][@"block_binding"] != nil) {
		[betas addObject:NFKAnthropicThinkingBindingBeta];
	}
	if (betas.count > 0) {
		[urlRequest setValue:[betas componentsJoinedByString:@","] forHTTPHeaderField:@"anthropic-beta"];
	}
	return urlRequest;
}

/*! Writes the key, as x-api-key or as a Bearer token when authorizesWithBearerToken is set, and the
	anthropic-version header, which a service that takes the Bearer token accepts too. */
- (void)authorizeURLRequest:(NSMutableURLRequest *)urlRequest
{
	NFKRemoteAPIStyle style = self.authorizesWithBearerToken ? NFKRemoteAPIStyleOpenAIChat : NFKRemoteAPIStyleAnthropicMessages;
	[NFKRemoteTransport authorizeRequest:urlRequest apiKey:self.apiKey style:style];
	[urlRequest setValue:self.apiVersion forHTTPHeaderField:@"anthropic-version"];
}

/*! The request's skills as container.skills, {type: anthropic | custom, skill_id, version}, with the
	code-execution tool the container runs them in when the request names no code-execution tool. An
	entry that carries a type passes through. */
- (void)addSkillsOf:(NFKInferenceRequest *)request toBody:(NSMutableDictionary *)body tools:(NSMutableArray *)tools
{
	NSArray *entries = [request.parameters[NFKParameterSkills] isKindOfClass:NSArray.class] ? request.parameters[NFKParameterSkills] : @[];
	NSMutableArray *skills = [NSMutableArray array];
	for (NSDictionary *entry in entries) {
		if (![entry isKindOfClass:NSDictionary.class]) {
			continue;
		}
		if (entry[@"type"] != nil) {
			[skills addObject:entry];
			continue;
		}
		if (![entry[@"id"] isKindOfClass:NSString.class]) {
			continue;
		}
		[skills addObject:@{ @"type": [entry[@"source"] isEqual:@"anthropic"] ? @"anthropic" : @"custom",
							 @"skill_id": entry[@"id"],
							 @"version": [entry[@"version"] isKindOfClass:NSString.class] ? entry[@"version"] : @"latest" }];
	}
	if (skills.count == 0) {
		return;
	}
	body[@"container"] = @{ @"skills": skills };
	for (NSDictionary *tool in tools) {
		if ([tool isKindOfClass:NSDictionary.class] && [tool[@"type"] isKindOfClass:NSString.class] && [tool[@"type"] hasPrefix:@"code_execution"]) {
			return;
		}
	}
	[tools addObject:@{ @"type": NFKAnthropicCodeExecutionToolType, @"name": @"code_execution" }];
}

/*! Builds the Messages request body: a system message lifted into the top-level field, media
	attached to the last user turn, tools in the API's shape, and a schema as a forced tool. */
- (nullable NSMutableDictionary *)bodyForRequest:(NFKInferenceRequest *)request attachments:(NFKRemoteAttachments *)attachments
									   maxTokens:(NSInteger)maxTokens
{
	NSMutableDictionary *body = [NSMutableDictionary dictionary];
	body[@"model"] = self.modelName;
	body[@"max_tokens"] = @(maxTokens);

	NSMutableDictionary *outputConfig = [NSMutableDictionary dictionary];
	BOOL thinks = [self addThinkingForEffort:[request parameterForKey:NFKParameterReasoningEffort]
									  toBody:body outputConfig:outputConfig];
	[self addPrefixMismatchSettingToBody:body];

	// Thinking rules the sampling: the API refuses temperature, top_p, or top_k beside it, and the
	// models from Claude Opus 4.7 on refuse them outright, so the three are dropped there. Meta's
	// route takes temperature and top_p beside the reasoning its models always do.
	BOOL samples = self.servesMuseModel || (!thinks && self.acceptsSampling);
	NSNumber *temperature = [request parameterForKey:NFKParameterTemperature];
	if ([temperature isKindOfClass:NSNumber.class] && samples) {
		body[@"temperature"] = temperature;
	}

	// The Messages API names these three itself, so the core keys are renamed rather than dropped.
	NSNumber *topP = [request parameterForKey:NFKParameterTopP];
	if ([topP isKindOfClass:NSNumber.class] && samples) {
		body[@"top_p"] = topP;
	}
	NSNumber *topK = [request parameterForKey:NFKParameterTopK];
	if ([topK isKindOfClass:NSNumber.class] && samples && !self.servesMuseModel) {
		body[@"top_k"] = topK;
	}
	NSArray<NSString *> *stopSequences = [request parameterForKey:NFKParameterStopSequences];
	if ([stopSequences isKindOfClass:NSArray.class] && !self.servesMuseModel) {
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
	if (!attachments.isEmpty && ![self attach:attachments toLastUserTurnIn:conversation citing:[[request parameterForKey:NFKParameterCitations] boolValue]]) {
		return nil;
	}
	body[@"messages"] = [self wireConversationFor:conversation];

	NSMutableArray *tools = [NSMutableArray array];
	NSArray *declared = request.parameters[NFKParameterTools];
	if ([declared isKindOfClass:NSArray.class]) {
		for (NSDictionary *tool in declared) {
			if ([tool isKindOfClass:NSDictionary.class]) {
				[tools addObject:[self wireToolForTool:tool]];
			}
		}
	}
	// Every model refuses a forced tool beside a thinking budget, so a budgeted request takes the
	// response format as well.
	BOOL forcesTool = self.acceptsForcedTool && !(thinks && self.usesThinkingBudget);
	NSDictionary *schema = request.parameters[NFKParameterJSONSchema];
	if ([schema isKindOfClass:NSDictionary.class] && forcesTool) {
		[tools addObject:@{ @"name": NFKAnthropicStructuredToolName,
							@"description": @"Record the structured reply.",
							@"input_schema": schema }];
		body[@"tool_choice"] = @{ @"type": @"tool", @"name": NFKAnthropicStructuredToolName };
	} else if ([schema isKindOfClass:NSDictionary.class]) {
		outputConfig[@"format"] = @{ @"type": @"json_schema", @"schema": schema };
	}
	[self addMCPServersOf:request toBody:body tools:tools];
	[self addSkillsOf:request toBody:body tools:tools];
	if (tools.count > 0) {
		body[@"tools"] = tools;
	}
	if (outputConfig.count > 0) {
		body[@"output_config"] = outputConfig;
	}
	return body;
}

/*! Writes the thinking a reasoning effort asks for and returns whether it wrote any. A model before
	Claude Opus 4.6 takes a token budget, which must leave room under max_tokens for the answer. A
	later model thinks adaptively at a named effort; max_tokens is raised by the same table, since
	the thinking counts toward it. From Claude Opus 4.7 on the thinking display defaults to omitted,
	so a summary is asked for to fill NFKOutputReasoning. */
- (BOOL)addThinkingForEffort:(nullable id)effort
					  toBody:(NSMutableDictionary *)body
				outputConfig:(NSMutableDictionary *)outputConfig
{
	if (effort == nil) {
		return NO;
	}
	NSNumber *budget = self.usesThinkingBudget
		? [self thinkingBudgetForEffort:effort]
		: (NFKAnthropicThinkingBudgets()[effort] ?: NFKAnthropicThinkingBudgets()[NFKReasoningEffortDeep]);
	if (self.usesThinkingBudget) {
		body[@"thinking"] = @{ @"type": @"enabled", @"budget_tokens": budget };
	} else if (NFKAnthropicModelIsInFamilies(self.modelName, NFKAnthropicSummarizingFamilies())) {
		body[@"thinking"] = @{ @"type": @"adaptive" };
		outputConfig[@"effort"] = [self effortLevelForEffort:effort];
	} else {
		body[@"thinking"] = @{ @"type": @"adaptive", @"display": @"summarized" };
		outputConfig[@"effort"] = [self effortLevelForEffort:effort];
	}
	if ([body[@"max_tokens"] integerValue] <= budget.integerValue) {
		body[@"max_tokens"] = @(budget.integerValue + (self.maxTokens > 0 ? self.maxTokens : NFKAnthropicFallbackMaxTokens));
	}
	return YES;
}

/*! Writes thinking.block_binding.prefix_mismatch_behavior for a model that runs the prefix check. The
	setting rides on an adaptive thinking object, which is what such a model runs when the request
	names none, so a request without an effort gains {type: adaptive} and thinks as it did. */
- (void)addPrefixMismatchSettingToBody:(NSMutableDictionary *)body
{
	NSString *setting = self.prefixMismatchSetting;
	if (setting == nil) {
		return;
	}
	NSMutableDictionary *thinking = [body[@"thinking"] mutableCopy] ?: [NSMutableDictionary dictionaryWithObject:@"adaptive" forKey:@"type"];
	thinking[@"block_binding"] = @{ @"prefix_mismatch_behavior": setting };
	body[@"thinking"] = thinking;
}

#pragma mark Model generations

// A name outside the known earlier families is taken as a current model: every model Anthropic has
// released since Claude Opus 4.6 refuses a thinking budget.
- (BOOL)usesThinkingBudget
{
	return NFKAnthropicModelIsInFamilies(self.modelName, NFKAnthropicBudgetThinkingFamilies());
}

- (BOOL)acceptsSampling
{
	return self.usesThinkingBudget || NFKAnthropicModelIsInFamilies(self.modelName, NFKAnthropicSummarizingFamilies());
}

- (BOOL)acceptsForcedTool
{
	return !self.servesMuseModel && !NFKAnthropicModelIsInFamilies(self.modelName, NFKAnthropicUnforcedToolFamilies());
}

- (BOOL)servesMuseModel
{
	return NFKAnthropicModelIsMuse(self.modelName);
}

- (nullable NSString *)prefixMismatchSetting
{
	if (!NFKAnthropicModelIsInFamilies(self.modelName, NFKAnthropicPrefixCheckFamilies())) {
		return nil;
	}
	switch (self.prefixMismatchBehavior) {
		case NFKAnthropicPrefixMismatchBehaviorError: return @"error";
		case NFKAnthropicPrefixMismatchBehaviorDropBlock: return @"drop_block";
		case NFKAnthropicPrefixMismatchBehaviorUnset: break;
	}
	return nil;
}

/*! The stop sequences the backend applies to the reply, for a model whose route refuses them on the
	wire. */
- (nullable NSArray<NSString *> *)clientStopsForRequest:(NFKInferenceRequest *)request
{
	NSArray<NSString *> *stops = [request parameterForKey:NFKParameterStopSequences];
	return self.servesMuseModel && [stops isKindOfClass:NSArray.class] && stops.count > 0 ? stops : nil;
}

// The three levels the contract names come from the table; a numeric string is a budget, which an
// adaptive model has no field for, so it is nil and refused. Any other string passes through, which
// is how a caller reaches xhigh or max.
- (nullable NSString *)effortLevelForEffort:(nullable id)effort
{
	if (![effort isKindOfClass:NSString.class] || [effort length] == 0) {
		return nil;
	}
	NSString *named = NFKAnthropicEffortLevels()[effort];
	if (named != nil) {
		return named;
	}
	NSScanner *scanner = [NSScanner scannerWithString:effort];
	NSInteger tokens = 0;
	if ([scanner scanInteger:&tokens] && scanner.isAtEnd) {
		return nil;
	}
	return effort;
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
// Claude's MCP connector names each server under mcp_servers and enables its tools through one
// mcp_toolset; an allowlist turns the rest off.
- (void)addMCPServersOf:(NFKInferenceRequest *)request toBody:(NSMutableDictionary *)body tools:(NSMutableArray *)tools
{
	NSArray *servers = [request parameterForKey:NFKParameterMCPServers];
	NSMutableArray *wire = [NSMutableArray array];
	for (NSDictionary *server in [servers isKindOfClass:NSArray.class] ? servers : @[]) {
		NSString *url = [server isKindOfClass:NSDictionary.class] ? [server[@"url"] description] : nil;
		NSString *name = [server isKindOfClass:NSDictionary.class] && [server[@"name"] isKindOfClass:NSString.class] ? server[@"name"] : nil;
		if (url.length == 0 || name.length == 0) {
			continue;
		}
		NSMutableDictionary *entry = [NSMutableDictionary dictionaryWithDictionary:@{ @"type": @"url", @"url": url, @"name": name }];
		entry[@"authorization_token"] = server[@"authorizationToken"];
		[wire addObject:entry];
		NSMutableDictionary *toolset = [NSMutableDictionary dictionaryWithDictionary:@{ @"type": @"mcp_toolset", @"mcp_server_name": name }];
		NSArray *allowed = [server[@"allowedTools"] isKindOfClass:NSArray.class] ? server[@"allowedTools"] : nil;
		if (allowed.count > 0) {
			toolset[@"default_config"] = @{ @"enabled": @NO };
			NSMutableDictionary *configs = [NSMutableDictionary dictionary];
			for (NSString *tool in allowed) {
				configs[tool] = @{ @"enabled": @YES };
			}
			toolset[@"configs"] = configs;
		}
		[tools addObject:toolset];
	}
	if (wire.count > 0) {
		body[@"mcp_servers"] = wire;
	}
}

// A server tool (web search, code execution) carries its type, and a tool already in the wire shape
// its input_schema; both pass through.
- (NSDictionary *)wireToolForTool:(NSDictionary *)tool
{
	if (tool[@"input_schema"] != nil || tool[@"type"] != nil) {
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
// frames as base64 image blocks, documents as base64 PDF document blocks. On Meta's route a video
// the service keeps is a video block that names its file.
// A PDF goes as base64; a plain-text document goes as its text, which the Messages API reads and cites
// by character range.
- (NSDictionary *)documentBlockFor:(NSDictionary *)document citing:(BOOL)citing
{
	NFKRemoteFile *file = document[@"fileReference"];
	if (file != nil && [file.mimeType hasPrefix:@"image/"]) {
		return @{ @"type": @"image", @"source": @{ @"type": @"file", @"file_id": file.identifier } };
	}
	if (file != nil && [file.mimeType hasPrefix:@"video/"] && self.servesMuseModel) {
		return @{ @"type": @"video", @"source": @{ @"type": @"file", @"file_id": file.identifier } };
	}
	NSMutableDictionary *block = [NSMutableDictionary dictionaryWithObjectsAndKeys:@"document", @"type", document[@"filename"], @"title", nil];
	if (file != nil) {
		block[@"source"] = @{ @"type": @"file", @"file_id": file.identifier };
	} else if ([document[@"mediaType"] isEqual:@"text/plain"]) {
		NSString *text = [[NSString alloc] initWithData:document[@"data"] encoding:NSUTF8StringEncoding] ?: @"";
		block[@"source"] = @{ @"type": @"text", @"media_type": @"text/plain", @"data": text };
	} else {
		block[@"source"] = @{ @"type": @"base64", @"media_type": @"application/pdf",
							  @"data": [document[@"data"] base64EncodedStringWithOptions:0] };
	}
	if (citing) {
		block[@"citations"] = @{ @"enabled": @YES };
	}
	return block;
}

- (BOOL)attach:(NFKRemoteAttachments *)attachments toLastUserTurnIn:(NSMutableArray *)conversation citing:(BOOL)citing
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
		if (attachments.videoData != nil) {
			NSString *format = [attachments.videoFormat isEqualToString:@"mov"] ? @"quicktime" : attachments.videoFormat ?: @"mp4";
			[blocks addObject:@{ @"type": @"video",
								 @"source": @{ @"type": @"base64", @"media_type": [@"video/" stringByAppendingString:format],
											   @"data": [attachments.videoData base64EncodedStringWithOptions:0] } }];
		}
		for (NSDictionary *document in attachments.documents) {
			[blocks addObject:[self documentBlockFor:document citing:citing]];
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

// The API carries a tool conversation as blocks: an assistant turn's calls are tool_use blocks, and
// the answers to one turn's calls are tool_result blocks opening the user turn that follows. An
// assistant turn this endpoint wrote replays its own blocks, thinking signatures included, which the
// API requires back while a tool is in use.
- (NSArray *)wireConversationFor:(NSArray *)conversation
{
	NSMutableArray *wire = [NSMutableArray arrayWithCapacity:conversation.count];
	NSMutableArray *results = [NSMutableArray array];
	for (NSDictionary *message in conversation) {
		if (NFKMessageIsToolResult(message)) {
			[results addObject:[self toolResultBlockFor:message]];
			continue;
		}
		NSDictionary *turn = [self wireTurnFor:message];
		if (results.count > 0 && [turn[@"role"] isEqual:@"user"]) {
			NSMutableArray *blocks = [results mutableCopy];
			id content = turn[@"content"];
			if ([content isKindOfClass:NSString.class] && [content length] > 0) {
				[blocks addObject:@{ @"type": @"text", @"text": content }];
			} else if ([content isKindOfClass:NSArray.class]) {
				[blocks addObjectsFromArray:content];
			}
			[wire addObject:@{ @"role": @"user", @"content": blocks }];
			[results removeAllObjects];
			continue;
		}
		if (results.count > 0) {
			[wire addObject:@{ @"role": @"user", @"content": [results copy] }];
			[results removeAllObjects];
		}
		[wire addObject:turn];
	}
	if (results.count > 0) {
		[wire addObject:@{ @"role": @"user", @"content": [results copy] }];
	}
	return wire;
}

- (NSDictionary *)wireTurnFor:(NSDictionary *)message
{
	if (![message isKindOfClass:NSDictionary.class] || ![message[@"role"] isEqual:@"assistant"]) {
		return NFKMessageWithoutMetadata(message);
	}
	id recorded = NFKMessageProviderTurn(message, NFKProviderTurnFormatAnthropicMessages, self.endpointURL.absoluteString);
	if ([recorded isKindOfClass:NSArray.class]) {
		return @{ @"role": @"assistant", @"content": recorded };
	}
	if (![message[@"tool_calls"] isKindOfClass:NSArray.class]) {
		return NFKMessageWithoutMetadata(message);
	}
	NSMutableArray *blocks = [NSMutableArray array];
	if ([message[@"content"] isKindOfClass:NSString.class] && [message[@"content"] length] > 0) {
		[blocks addObject:@{ @"type": @"text", @"text": message[@"content"] }];
	}
	for (NSDictionary *call in [NFKToolMessages toolCallsOfAssistantMessage:message]) {
		[blocks addObject:@{ @"type": @"tool_use", @"id": call[@"id"], @"name": call[@"name"], @"input": call[@"arguments"] }];
	}
	return @{ @"role": @"assistant", @"content": blocks };
}

- (NSDictionary *)toolResultBlockFor:(NSDictionary *)message
{
	NSMutableDictionary *block = [NSMutableDictionary dictionaryWithDictionary:@{ @"type": @"tool_result",
																				 @"tool_use_id": NFKToolMessageCallIdentifier(message) }];
	NSString *text = NFKToolMessageText(message);
	NSArray<NSData *> *images = NFKToolMessageImagePNGs(message);
	if (images.count == 0) {
		block[@"content"] = text;
	} else {
		NSMutableArray *content = [NSMutableArray array];
		if (text.length > 0) {
			[content addObject:@{ @"type": @"text", @"text": text }];
		}
		for (NSData *png in images) {
			[content addObject:@{ @"type": @"image",
								  @"source": @{ @"type": @"base64", @"media_type": @"image/png",
												@"data": [png base64EncodedStringWithOptions:0] } }];
		}
		block[@"content"] = content;
	}
	if (NFKToolMessageIsError(message)) {
		block[@"is_error"] = @YES;
	}
	return block;
}

- (BOOL)requestExpectsStructuredReply:(NFKInferenceRequest *)request
{
	return [request.parameters[NFKParameterJSONSchema] isKindOfClass:NSDictionary.class];
}

#pragma mark Response

// The API splits the input three ways: input_tokens is only what was read past the last cache
// breakpoint, beside cache_read_input_tokens and cache_creation_input_tokens. NFKUsageInputTokens is the
// whole input, so it is their sum. Anthropic reports no reasoning count: the thinking tokens are part of
// the output total. Meta's route reports it under output_tokens_details.thinking_tokens.
- (nullable NSDictionary<NSString *, NSNumber *> *)usageInMessageBody:(nullable NSDictionary *)body
{
	NSDictionary *usage = [body[@"usage"] isKindOfClass:NSDictionary.class] ? body[@"usage"] : nil;
	if (usage == nil) {
		return nil;
	}
	NSNumber *uncached = [usage[@"input_tokens"] isKindOfClass:NSNumber.class] ? usage[@"input_tokens"] : nil;
	NSNumber *cacheRead = [usage[@"cache_read_input_tokens"] isKindOfClass:NSNumber.class] ? usage[@"cache_read_input_tokens"] : nil;
	NSNumber *cacheWrite = [usage[@"cache_creation_input_tokens"] isKindOfClass:NSNumber.class] ? usage[@"cache_creation_input_tokens"] : nil;
	NSNumber *input = nil;
	if (uncached != nil || cacheRead != nil || cacheWrite != nil) {
		input = @(uncached.integerValue + cacheRead.integerValue + cacheWrite.integerValue);
	}
	NSDictionary *outputDetails = [usage[@"output_tokens_details"] isKindOfClass:NSDictionary.class] ? usage[@"output_tokens_details"] : nil;
	NSNumber *thinking = [outputDetails[@"thinking_tokens"] isKindOfClass:NSNumber.class] ? outputDetails[@"thinking_tokens"] : nil;
	return NFKRemoteUsage(input, cacheRead, usage[@"output_tokens"], thinking);
}

/*! Joins the text blocks into the text output and reads the tool uses. The API returns a list of
	typed blocks, not one string; the forced structured tool's input is the structured output. The
	text ends at the earliest stop sequence the route left to the backend, and the finish reason is
	the reply's stop reason, or a stop sequence where one cut the text. */
- (NFKInferenceResult *)resultForContentBlocks:(NSArray *)content
											 usage:(nullable NSDictionary<NSString *, NSNumber *> *)usage
										   raw:(id)raw
									stopReason:(nullable id)stopReason
							   transformations:(nullable NSArray<NSDictionary *> *)transformations
									   request:(NFKInferenceRequest *)request
{
	BOOL expectsStructured = [self requestExpectsStructuredReply:request];
	id finishReason = stopReason;
	NSMutableDictionary *outputs = [NSMutableDictionary dictionary];
	NSMutableArray<NSString *> *pieces = [NSMutableArray array];
	NSMutableArray<NSString *> *reasoning = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *toolCalls = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *citations = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *serverCalls = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *serverResults = [NSMutableArray array];
	NSMutableArray *turn = [NSMutableArray array];
	for (NSDictionary *block in content) {
		BOOL structuredBlock = expectsStructured && [block isKindOfClass:NSDictionary.class]
			&& [block[@"type"] isEqual:@"tool_use"] && [block[@"name"] isEqual:NFKAnthropicStructuredToolName];
		if (!structuredBlock) {
			[turn addObject:block];
		}
	}
	for (NSDictionary *block in content) {
		if (![block isKindOfClass:NSDictionary.class]) {
			continue;
		}
		if ([block[@"type"] isEqualToString:@"text"] && [block[@"text"] isKindOfClass:NSString.class]) {
			[pieces addObject:block[@"text"]];
			[self addCitationsOfTextBlock:block to:citations];
			continue;
		}
		if ([block[@"type"] isEqualToString:@"server_tool_use"] || [block[@"type"] isEqualToString:@"mcp_tool_use"]) {
			[serverCalls addObject:block];
			continue;
		}
		if ([block[@"type"] hasSuffix:@"_tool_result"]) {
			[serverResults addObject:block];
			continue;
		}
		// A thinking block carries the chain; a redacted one carries encrypted bytes with nothing
		// to read, and one whose display was omitted carries an empty string, so neither
		// contributes reasoning text.
		if ([block[@"type"] isEqualToString:@"thinking"]) {
			if ([block[@"thinking"] isKindOfClass:NSString.class] && [block[@"thinking"] length] > 0) {
				[reasoning addObject:block[@"thinking"]];
			}
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
		NSString *text = [pieces componentsJoinedByString:@""];
		NSArray<NSString *> *clientStops = [self clientStopsForRequest:request];
		NSString *kept = clientStops != nil ? NFKRemoteTextBeforeStops(text, clientStops) : text;
		if (kept.length < text.length) {
			finishReason = NFKFinishReasonStopSequence;
		}
		outputs[NFKRemoteBackendTextKey] = kept;
	}
	// A schema asked for through output_config.format comes back as the text of the reply.
	if (expectsStructured && outputs[NFKOutputStructured] == nil && pieces.count > 0) {
		NSData *text = [outputs[NFKRemoteBackendTextKey] dataUsingEncoding:NSUTF8StringEncoding];
		id structured = [NSJSONSerialization JSONObjectWithData:text options:0 error:NULL];
		if ([structured isKindOfClass:NSDictionary.class]) {
			outputs[NFKOutputStructured] = structured;
		}
	}
	if (reasoning.count > 0) {
		outputs[NFKOutputReasoning] = [reasoning componentsJoinedByString:@"\n"];
	}
	if (toolCalls.count > 0) {
		outputs[NFKOutputToolCalls] = toolCalls;
	}
	outputs[NFKOutputAssistantMessage] = NFKAssistantMessage(outputs[NFKRemoteBackendTextKey], toolCalls,
															 NFKProviderTurnFormatAnthropicMessages,
															 self.endpointURL.absoluteString, turn);
	if (usage != nil) {
		outputs[NFKOutputUsage] = usage;
	}
	if (citations.count > 0) {
		outputs[NFKOutputCitations] = citations;
	}
	NSArray<NSDictionary *> *ran = [self serverToolResultsPairing:serverCalls with:serverResults];
	if (ran.count > 0) {
		outputs[NFKOutputServerToolResults] = ran;
	}
	if (transformations.count > 0) {
		outputs[NFKAnthropicBackendInputTransformationsKey] = transformations;
	}
	outputs[NFKRemoteBackendRawKey] = raw;
	return NFKResultWithFinishReason([NFKInferenceResult resultWithOutputs:outputs], finishReason);
}

// A text block's citations name the span they support, as a character range, a page range, a block
// range, or a search result, beside the cited text itself.
- (void)addCitationsOfTextBlock:(NSDictionary *)block to:(NSMutableArray<NSDictionary *> *)citations
{
	NSArray *entries = [block[@"citations"] isKindOfClass:NSArray.class] ? block[@"citations"] : @[];
	for (NSDictionary *entry in entries) {
		if (![entry isKindOfClass:NSDictionary.class]) {
			continue;
		}
		NSMutableDictionary *citation = [NSMutableDictionary dictionaryWithObject:entry forKey:@"raw"];
		citation[@"text"] = entry[@"cited_text"];
		citation[@"url"] = entry[@"url"];
		citation[@"title"] = entry[@"title"] ?: entry[@"document_title"];
		citation[@"documentIndex"] = entry[@"document_index"] ?: entry[@"search_result_index"];
		citation[@"start"] = entry[@"start_char_index"] ?: entry[@"start_page_number"] ?: entry[@"start_block_index"];
		citation[@"end"] = entry[@"end_char_index"] ?: entry[@"end_page_number"] ?: entry[@"end_block_index"];
		[citations addObject:citation];
	}
}

// A server tool's call and its result are separate blocks joined by the call's id.
- (NSArray<NSDictionary *> *)serverToolResultsPairing:(NSArray<NSDictionary *> *)calls with:(NSArray<NSDictionary *> *)results
{
	NSMutableArray<NSDictionary *> *ran = [NSMutableArray array];
	for (NSDictionary *call in calls) {
		NSMutableDictionary *entry = [NSMutableDictionary dictionaryWithObject:call forKey:@"raw"];
		entry[@"name"] = call[@"name"];
		entry[@"input"] = call[@"input"];
		for (NSDictionary *result in results) {
			if ([result[@"tool_use_id"] isEqual:call[@"id"]]) {
				entry[@"output"] = result[@"content"];
				break;
			}
		}
		[ran addObject:entry];
	}
	return ran;
}

#pragma mark Streaming

// content_block_start opens a block; content_block_delta appends text or partial JSON to it.
// Returns whether the event carried text, which is what a partial result is worth reporting for.
- (BOOL)applyStreamEvent:(NSDictionary *)event type:(NSString *)type toState:(NFKAnthropicStreamState *)state
{
	// message_start opens with what the request cost; message_delta closes with what the reply cost,
	// so the two together are the turn's counts.
	if ([type isEqualToString:@"message_delta"]) {
		NSDictionary *delta = [event[@"delta"] isKindOfClass:NSDictionary.class] ? event[@"delta"] : nil;
		if ([delta[@"stop_reason"] isKindOfClass:NSString.class]) {
			state.stopReason = delta[@"stop_reason"];
		}
	}
	if ([type isEqualToString:@"message_start"] || [type isEqualToString:@"message_delta"]) {
		NSDictionary *carrier = [event[@"message"] isKindOfClass:NSDictionary.class] ? event[@"message"] : event;
		if ([carrier[@"input_transformations"] isKindOfClass:NSArray.class]) {
			state.inputTransformations = carrier[@"input_transformations"];
		}
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
	if ([delta[@"type"] isEqual:@"citations_delta"] && [delta[@"citation"] isKindOfClass:NSDictionary.class]) {
		NSArray *existing = [block[@"citations"] isKindOfClass:NSArray.class] ? block[@"citations"] : @[];
		block[@"citations"] = [existing arrayByAddingObject:delta[@"citation"]];
		return NO;
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
		if ([block[@"type"] isEqual:type] && [block[type] isKindOfClass:NSString.class] && [block[type] length] > 0) {
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
- (NFKInferenceResult *)resultFromStreamState:(NFKAnthropicStreamState *)state request:(NFKInferenceRequest *)request
{
	NSMutableArray *content = [NSMutableArray array];
	for (NSNumber *index in [state.blocksByIndex.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		NSMutableDictionary *block = [state.blocksByIndex[index] mutableCopy];
		if ([block[@"type"] isEqual:@"tool_use"] || [block[@"type"] isEqual:@"server_tool_use"]) {
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
							 stopReason:state.stopReason
						transformations:[self transformationsAfter:state.removedThinking reported:state.inputTransformations]
								request:request];
}

#pragma mark Prefix check

/*! Whether the error is the 400 a replayed thinking block gets when the content before it changed
	since the model produced it. */
- (BOOL)isPrefixMismatchError:(nullable NSError *)error
{
	return [error.userInfo[NFKRemoteErrorStatusCodeKey] integerValue] == 400
		&& [error.userInfo[NFKRemoteErrorBodyKey] containsString:NFKAnthropicPrefixMismatchSentence];
}

/*! The position the error names, messages.{i}.content.{j}, as {i, j}, or nil where it names none. The
	message opens with the position, so the first match is the failed block's. */
- (nullable NSArray<NSNumber *> *)blockPositionNamedInError:(NSError *)error
{
	NSString *detail = error.userInfo[NFKRemoteErrorBodyKey] ?: @"";
	NSRegularExpression *path = [NSRegularExpression regularExpressionWithPattern:@"messages\\.(\\d+)\\.content\\.(\\d+)" options:0 error:NULL];
	NSTextCheckingResult *match = [path firstMatchInString:detail options:0 range:NSMakeRange(0, detail.length)];
	if (match == nil) {
		return nil;
	}
	return @[ @([detail substringWithRange:[match rangeAtIndex:1]].integerValue),
			  @([detail substringWithRange:[match rangeAtIndex:2]].integerValue) ];
}

/*! Whether two neighbouring messages share a role or a message has the system role. The error counts
	such a message as part of the one before it, so the position it names is not an index into the
	array. */
- (BOOL)errorPositionsAreAmbiguousFor:(NSArray *)messages
{
	NSString *previous = nil;
	for (NSDictionary *message in messages) {
		NSString *role = [message isKindOfClass:NSDictionary.class] && [message[@"role"] isKindOfClass:NSString.class] ? message[@"role"] : @"";
		if ([role isEqualToString:@"system"] || [role isEqualToString:previous]) {
			return YES;
		}
		previous = role;
	}
	return NO;
}

/*! urlRequest without the thinking the prefix check refused, or nil where failure is another error,
	recoversFromPrefixMismatch is off, or the body holds no thinking to remove. The recovery is the one
	Anthropic documents for removing the blocks by hand: the named block and every thinking and
	redacted_thinking block after it go, and every thinking block goes where the named position cannot be
	read against the array. Each removed block is listed in outRemoved as {type: thinking_removed, path},
	the path being its place in the request as first sent. */
- (nullable NSMutableURLRequest *)requestByRemovingMismatchedThinkingFrom:(NSURLRequest *)urlRequest
															   afterError:(nullable NSError *)failure
																  removed:(NSArray<NSDictionary *> * _Nullable * _Nullable)outRemoved
{
	if (!self.recoversFromPrefixMismatch || ![self isPrefixMismatchError:failure] || urlRequest.HTTPBody == nil) {
		return nil;
	}
	NSMutableDictionary *body = [NSJSONSerialization JSONObjectWithData:urlRequest.HTTPBody options:NSJSONReadingMutableContainers error:NULL];
	NSMutableArray *messages = [body isKindOfClass:NSDictionary.class] && [body[@"messages"] isKindOfClass:NSArray.class] ? body[@"messages"] : nil;
	if (messages == nil) {
		return nil;
	}
	NSArray<NSNumber *> *named = [self errorPositionsAreAmbiguousFor:messages] ? nil : [self blockPositionNamedInError:failure];
	NSInteger firstMessage = named != nil ? named[0].integerValue : 0;
	NSInteger firstBlock = named != nil ? named[1].integerValue : 0;
	NSMutableArray<NSDictionary *> *removed = [NSMutableArray array];
	for (NSInteger index = firstMessage; index < (NSInteger)messages.count; index++) {
		NSMutableDictionary *message = [messages[index] isKindOfClass:NSDictionary.class] ? messages[index] : nil;
		NSArray *content = [message[@"content"] isKindOfClass:NSArray.class] ? message[@"content"] : nil;
		NSMutableArray *kept = [NSMutableArray array];
		for (NSInteger position = 0; position < (NSInteger)content.count; position++) {
			NSDictionary *block = content[position];
			BOOL thinking = [block isKindOfClass:NSDictionary.class]
				&& ([block[@"type"] isEqual:@"thinking"] || [block[@"type"] isEqual:@"redacted_thinking"]);
			if (thinking && (index > firstMessage || position >= firstBlock)) {
				[removed addObject:@{ @"type": @"thinking_removed",
									  @"path": [NSString stringWithFormat:@"messages.%ld.content.%ld", (long)index, (long)position] }];
				continue;
			}
			[kept addObject:block];
		}
		if (content != nil) {
			message[@"content"] = kept;
		}
	}
	if (removed.count == 0) {
		return nil;
	}
	NSMutableURLRequest *retry = [urlRequest mutableCopy];
	retry.HTTPBody = [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL];
	if (outRemoved != NULL) {
		*outRemoved = removed;
	}
	return retry;
}

/*! Sends urlRequest and returns the body of a 2xx answer, or nil with the transport's or the status's
	error. A request the prefix check refuses is sent once more without the mismatched thinking, and
	outRemoved then lists the removed blocks. */
- (nullable NSData *)dataBySendingRequest:(NSURLRequest *)urlRequest
						  removedThinking:(NSArray<NSDictionary *> * _Nullable * _Nullable)outRemoved
									error:(NSError * _Nullable *)outError
{
	NSURLRequest *sending = urlRequest;
	for (NSInteger attempt = 0; ; attempt++) {
		NSHTTPURLResponse *response = nil;
		NSError *sendError = nil;
		NSData *data = [self sendRequest:sending response:&response error:&sendError];
		NSError *failure = data == nil
			? (sendError ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the call returned nothing"])
			: [NFKRemoteTransport errorForResponse:response data:data];
		if (failure == nil) {
			return data;
		}
		NSURLRequest *retry = attempt == 0 ? [self requestByRemovingMismatchedThinkingFrom:sending afterError:failure removed:outRemoved] : nil;
		if (retry == nil) {
			if (outError != NULL) { *outError = failure; }
			return nil;
		}
		sending = retry;
	}
}

/*! The blocks the backend removed, then the input_transformations entries the reply reports under the
	thinking-binding beta. */
- (nullable NSArray<NSDictionary *> *)transformationsAfter:(nullable NSArray<NSDictionary *> *)removedThinking
												  reported:(nullable NSArray *)reported
{
	NSArray<NSDictionary *> *all = [removedThinking ?: @[] arrayByAddingObjectsFromArray:reported ?: @[]];
	return all.count > 0 ? all : nil;
}

- (nullable NSArray<NSDictionary *> *)transformationsAfter:(nullable NSArray<NSDictionary *> *)removedThinking
												reportedIn:(nullable NSDictionary *)reply
{
	NSArray *reported = [reply[@"input_transformations"] isKindOfClass:NSArray.class] ? reply[@"input_transformations"] : nil;
	return [self transformationsAfter:removedThinking reported:reported];
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

#pragma mark - Batches

@implementation NFKAnthropicBackend (NFKRemoteBatch)

- (nullable NSString *)batchModelName
{
	return self.modelName;
}

// A batch never asks the Models API for the model's maximum, so a request without its own limit takes
// the fallback limit.
- (nullable NSURLRequest *)batchURLRequestForRequest:(NFKInferenceRequest *)request error:(NSError * _Nullable *)outError
{
	NSInteger maxTokens = [self maxTokensForRequest:request lookingUp:NO modelMaximum:NULL];
	return [self urlRequestForRequest:request streaming:NO maxTokens:maxTokens modelMaximum:NO error:outError];
}

- (nullable NFKInferenceResult *)batchResultForReply:(NSDictionary *)reply
											 request:(NFKInferenceRequest *)request
											   error:(NSError * _Nullable *)outError
{
	if ([reply[@"stop_reason"] isEqual:@"refusal"]) {
		return [self failWithCode:kNFKError_InferenceRefused description:@"the model declined to answer" error:outError];
	}
	NSArray *content = [reply[@"content"] isKindOfClass:NSArray.class] ? reply[@"content"] : @[];
	return [self resultForContentBlocks:content
								  usage:[self usageInMessageBody:reply]
									raw:reply
							 stopReason:reply[@"stop_reason"]
						transformations:[self transformationsAfter:nil reportedIn:reply]
								request:request];
}

@end
