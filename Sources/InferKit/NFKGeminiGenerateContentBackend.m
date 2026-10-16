//
//  NFKGeminiGenerateContentBackend.m
//  InferKit
//

#import <InferKit/NFKGeminiGenerateContentBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKRemoteFileStore.h>
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
#import "NFKJobGuard.h"

/*! The contract keys this backend translates or has no field for; every other parameter goes into
	generationConfig, or the request's top level for the names in NFKGenerateContentTopLevelParameters. */
static NSSet<NSString *> *NFKGenerateContentMappedParameters(void)
{
	static NSSet<NSString *> *mapped;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		mapped = [NSSet setWithArray:@[ NFKParameterTools, NFKParameterJSONSchema, NFKParameterReasoningEffort,
										NFKParameterMaxTokens, NFKParameterTemperature, NFKParameterTopP,
										NFKParameterTopK, NFKParameterSeed, NFKParameterStopSequences,
										NFKParameterRepetitionPenalty, NFKParameterConversationKey,
										NFKParameterMCPServers, NFKParameterCitations, NFKParameterOutputFormat,
										NFKParameterPreviousResponseIdentifier, NFKParameterVideoFrameCount ]];
	});
	return mapped;
}

static NSSet<NSString *> *NFKGenerateContentTopLevelParameters(void)
{
	static NSSet<NSString *> *topLevel;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		topLevel = [NSSet setWithArray:@[ @"safetySettings", @"toolConfig", @"cachedContent", @"labels",
										  @"serviceTier", @"store" ]];
	});
	return topLevel;
}

/*! The finish reasons that mean the service withheld the reply. */
static NSSet<NSString *> *NFKGenerateContentRefusingFinishReasons(void)
{
	static NSSet<NSString *> *reasons;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		reasons = [NSSet setWithArray:@[ @"SAFETY", @"RECITATION", @"BLOCKLIST", @"PROHIBITED_CONTENT", @"SPII",
										 @"IMAGE_SAFETY", @"IMAGE_PROHIBITED_CONTENT", @"IMAGE_RECITATION" ]];
	});
	return reasons;
}

// A call the service returned without an id gets one of these, and the answer to it goes back
// without an id, as the service matches it by name.
static NSString * const NFKGenerateContentGeneratedCallPrefix = @"gemini-call-";

static id _Nullable NFKGenerateContentValue(id _Nullable value, Class kind)
{
	return [value isKindOfClass:kind] ? value : nil;
}

/*! The text of a byte range of a UTF-8 string, which is how the API measures citation spans. */
static NSString * _Nullable NFKGenerateContentTextInBytes(NSString *text, id start, id end)
{
	if (![start isKindOfClass:NSNumber.class] || ![end isKindOfClass:NSNumber.class]) {
		return nil;
	}
	NSData *bytes = [text dataUsingEncoding:NSUTF8StringEncoding];
	NSUInteger from = [start unsignedIntegerValue], to = [end unsignedIntegerValue];
	if (from >= to || to > bytes.length) {
		return nil;
	}
	return [[NSString alloc] initWithData:[bytes subdataWithRange:NSMakeRange(from, to - from)] encoding:NSUTF8StringEncoding];
}

/*! What a stream has delivered so far, merged into the shape of one reply. */
@interface NFKGenerateContentStreamState : NSObject
@property (nonatomic, strong) NSMutableArray<NSMutableDictionary *> *parts;
@property (nonatomic, strong) NSMutableDictionary *candidate;
@property (nonatomic, strong) NSMutableDictionary *reply;
@property (nonatomic, copy, nullable) NSString *failure;
@property (nonatomic, strong) NSMutableSet<NSString *> *continuationTokens;
@end

@implementation NFKGenerateContentStreamState
- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_parts = [NSMutableArray array];
		_candidate = [NSMutableDictionary dictionary];
		_reply = [NSMutableDictionary dictionary];
		_continuationTokens = [NSMutableSet set];
	}
	return self;
}
@end

/*! The most hops one reply follows. A long decode's hops are tens of thousands of tokens each, so a reply
	still asking for more after this many is a service that never finishes. */
static const NSUInteger NFKGenerateContentHopLimit = 64;

@implementation NFKGeminiGenerateContentBackend

@synthesize session = _session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKGeminiGenerateContentBackend *backend = [[self alloc] init];
	if (endpointURL != nil) {
		backend.endpointURL = endpointURL;
	}
	return backend;
}

+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
{
	NSURL *endpointURL = nil;
	if ([provider.identifier isEqualToString:@"gemini"]) {
		// The preset's base is the OpenAI layer (…/v1beta/openai); the native routes sit beside it.
		endpointURL = [provider.baseURL.URLByDeletingLastPathComponent URLByAppendingPathComponent:@"models"];
	} else if ([provider.modelAPIStyles.allValues containsObject:@(NFKRemoteAPIStyleGeminiGenerateContent)]) {
		endpointURL = [provider URLForPath:@"models"];
	}
	if (endpointURL == nil) {
		return nil;
	}
	NFKGeminiGenerateContentBackend *backend = [self backendWithEndpointURL:endpointURL];
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
		_endpointURL = [NSURL URLWithString:@"https://generativelanguage.googleapis.com/v1beta/models"];
		_timeout = 600.0;
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
	return @"gemini-generatecontent";
}

- (NSSet<NSString *> *)supportedOutputKeys
{
	return [NSSet setWithArray:@[ NFKOutputText, NFKOutputStructured, NFKOutputReasoning, NFKOutputImage,
								  NFKOutputImages, NFKOutputToolCalls, NFKOutputServerToolResults,
								  NFKOutputCitations, NFKOutputUsage, NFKOutputAssistantMessage,
								  NFKRemoteBackendRawKey ]];
}

// A reply that stops with finishReason CONTINUATION (a long decode, gemini-4-argon) is continued by
// resending the same request with its continuationToken; the hops' parts join into one reply.
- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError * _Nullable *)outError
{
	NSURLRequest *urlRequest = [self urlRequestForRequest:request streaming:NO error:outError];
	if (urlRequest == nil) {
		return nil;
	}
	NSDictionary *reply = [self replyForRequest:urlRequest error:outError];
	if (reply == nil) {
		return nil;
	}
	if (![self replyAsksToContinue:reply]) {
		return [self resultFromReply:reply request:request error:outError];
	}
	NFKGenerateContentStreamState *state = [[NFKGenerateContentStreamState alloc] init];
	[self applyChunk:reply toState:state];
	while ([self replyAsksToContinue:reply]) {
		urlRequest = [self request:urlRequest continuingReply:reply state:state error:outError];
		reply = urlRequest != nil ? [self replyForRequest:urlRequest error:outError] : nil;
		if (reply == nil) {
			return nil;
		}
		[self applyChunk:reply toState:state];
		if (state.failure != nil) {
			return [self fail:outError code:kNFKError_InferenceBackendFailure reason:state.failure];
		}
	}
	return [self resultFromReply:[self replyOfState:state] request:request error:outError];
}

- (nullable NSDictionary *)replyForRequest:(NSURLRequest *)urlRequest error:(NSError * _Nullable *)outError
{
	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:urlRequest response:&response error:&sendError];
	NSError *failure = data == nil ? sendError : [NFKRemoteTransport errorForResponse:response data:data];
	if (failure != nil || data == nil) {
		if (outError != NULL) {
			*outError = failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the request failed"];
		}
		return nil;
	}
	NSDictionary *reply = NFKGenerateContentValue([NSJSONSerialization JSONObjectWithData:data options:0 error:NULL], NSDictionary.class);
	return reply ?: [self fail:outError code:kNFKError_InferenceBackendFailure reason:@"the response is not a JSON object"];
}

/*! Whether the reply stops at a hop boundary (finishReason CONTINUATION). A reply cut at
	maxOutputTokens also carries a token, which is not followed: the caller's limit stands. */
- (BOOL)replyAsksToContinue:(NSDictionary *)reply
{
	NSDictionary *candidate = NFKGenerateContentValue([NFKGenerateContentValue(reply[@"candidates"], NSArray.class) firstObject], NSDictionary.class);
	return [candidate[@"finishReason"] isEqual:@"CONTINUATION"];
}

/*!
	The same request, contents and generationConfig unchanged, carrying the reply's continuation token;
	the joined state drops the hop's token and finish reason, which the next hop replaces. Fails with kNFKError_InferenceBackendFailure, so a cut reply never reads as a whole one, when:
	- the reply has no token
	- the token is one an earlier hop already followed
	- the reply has followed NFKGenerateContentHopLimit hops
*/
- (nullable NSURLRequest *)request:(NSURLRequest *)urlRequest
				   continuingReply:(NSDictionary *)reply
							 state:(NFKGenerateContentStreamState *)state
							 error:(NSError * _Nullable *)outError
{
	NSString *token = NFKGenerateContentValue(reply[@"continuationToken"], NSString.class);
	if (token.length == 0) {
		return [self fail:outError code:kNFKError_InferenceBackendFailure reason:@"the reply stopped for continuation without a continuation token"];
	}
	if ([state.continuationTokens containsObject:token]) {
		return [self fail:outError code:kNFKError_InferenceBackendFailure reason:@"the service repeated a continuation token"];
	}
	if (state.continuationTokens.count >= NFKGenerateContentHopLimit) {
		return [self fail:outError code:kNFKError_InferenceBackendFailure
				   reason:[NSString stringWithFormat:@"the reply was still continuing after %lu hops", (unsigned long)NFKGenerateContentHopLimit]];
	}
	[state.continuationTokens addObject:token];
	NSMutableDictionary *body = [NFKGenerateContentValue([NSJSONSerialization JSONObjectWithData:urlRequest.HTTPBody ?: [NSData data]
																						 options:NSJSONReadingMutableContainers error:NULL],
														 NSDictionary.class) mutableCopy];
	body[@"continuationToken"] = token;
	NSData *payload = body != nil ? [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL] : nil;
	if (payload == nil) {
		return [self fail:outError code:kNFKError_InferenceBackendFailure reason:@"the request could not be continued"];
	}
	[state.reply removeObjectForKey:@"continuationToken"];
	[state.candidate removeObjectForKey:@"finishReason"];
	NSMutableURLRequest *next = [urlRequest mutableCopy];
	next.HTTPBody = payload;
	return next;
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	NSError *error = nil;
	NSMutableURLRequest *urlRequest = [self urlRequestForRequest:request streaming:YES error:&error];
	if (urlRequest == nil) {
		[job finishWithError:error ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the request failed"]];
		return job;
	}
	[job reportProgress:-1.0];
	[self streamHop:urlRequest state:[[NFKGenerateContentStreamState alloc] init] job:job request:request];
	return job;
}

// One stream; a reply cut at a hop boundary opens the next stream with its continuation token, the
// parts and the partial text carrying on in the same state.
- (void)streamHop:(NSURLRequest *)urlRequest state:(NFKGenerateContentStreamState *)state job:(NFKInferenceJob *)job request:(NFKInferenceRequest *)request
{
	__block BOOL finished = NO;
	void (^cancel)(void) = [self streamRequest:urlRequest lineHandler:^(NSString *line) {
		NSString *payload = [NFKRemoteTransport SSEDataForLine:line];
		NSDictionary *chunk = payload != nil
			? NFKGenerateContentValue([NSJSONSerialization JSONObjectWithData:[payload dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL], NSDictionary.class)
			: nil;
		if (finished || chunk == nil) {
			return;
		}
		if ([self applyChunk:chunk toState:state]) {
			[job reportProgress:-1.0 partialResult:[NFKInferenceResult resultWithOutputs:@{ NFKOutputText: [self textOfParts:state.parts] }]];
		}
	} completionHandler:^(NSHTTPURLResponse * _Nullable response, NSData * _Nullable errorBody, NSError * _Nullable streamError) {
		NFKJobRunGuarded(job, ^{
			if (finished || job.status == NFKInferenceJobStatusCancelled) {
				return;
			}
			finished = YES;
			NSError *failure = streamError ?: [NFKRemoteTransport errorForResponse:response data:errorBody];
			if (failure == nil && state.failure != nil) {
				failure = [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:state.failure];
			}
			NSDictionary *reply = [self replyOfState:state];
			if (failure == nil && [self replyAsksToContinue:reply]) {
				NSURLRequest *next = [self request:urlRequest continuingReply:reply state:state error:&failure];
				if (next != nil) {
					[self streamHop:next state:state job:job request:request];
					return;
				}
			}
			NFKInferenceResult *result = failure == nil ? [self resultFromReply:reply request:request error:&failure] : nil;
			if (result != nil) {
				[job finishWithResult:result];
			} else {
				[job finishWithError:failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the stream failed"]];
			}
		});
	}];
	job.cancellationHandler = cancel;
	// A cancel that arrived while the previous hop handed over called that hop's finished handler.
	if (job.status == NFKInferenceJobStatusCancelled) {
		cancel();
	}
}

#pragma mark Request

- (nullable NSMutableURLRequest *)urlRequestForRequest:(NFKInferenceRequest *)request
											 streaming:(BOOL)streaming
												 error:(NSError * _Nullable *)outError
{
	if (self.modelName.length == 0) {
		return [self fail:outError code:kNFKError_InferenceNotReady reason:@"generateContent needs a model name"];
	}
	NFKRemoteAttachments *attachments = [NFKRemoteAttachments attachmentsForRequest:request keepsVideo:YES error:outError];
	if (attachments == nil) {
		return nil;
	}
	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
	NSArray *contents = [self contentsForRequest:request attachments:attachments body:body];
	if (contents == nil) {
		return [self fail:outError code:kNFKError_InferenceMissingInput reason:@"the request carries neither a prompt, messages, nor media"];
	}
	body[@"contents"] = contents;
	[self addConfigurationForRequest:request to:body];
	NSError *encodeError = nil;
	NSData *payload = [NSJSONSerialization dataWithJSONObject:body options:0 error:&encodeError];
	if (payload == nil) {
		if (outError != NULL) {
			*outError = encodeError;
		}
		return nil;
	}
	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:[self URLForMethod:streaming ? @"streamGenerateContent" : @"generateContent"]];
	urlRequest.HTTPMethod = @"POST";
	urlRequest.timeoutInterval = self.timeout;
	urlRequest.HTTPBody = payload;
	[urlRequest setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	if (self.apiKey.length > 0) {
		[urlRequest setValue:self.apiKey forHTTPHeaderField:@"x-goog-api-key"];
	}
	return urlRequest;
}

// The model is a path segment ending in the method (…/models/gemini-3.8-flash:generateContent); a
// name already written as models/… is not doubled.
- (NSURL *)URLForMethod:(NSString *)method
{
	NSString *model = [self.modelName hasPrefix:@"models/"] ? [self.modelName substringFromIndex:7] : self.modelName;
	NSCharacterSet *slash = [NSCharacterSet characterSetWithCharactersInString:@"/"];
	NSString *base = [self.endpointURL.absoluteString stringByTrimmingCharactersInSet:slash];
	NSString *suffix = [method isEqualToString:@"streamGenerateContent"] ? @"?alt=sse" : @"";
	return [NSURL URLWithString:[NSString stringWithFormat:@"%@/%@:%@%@", base, model, method, suffix]];
}

// Each turn is a Content of parts. The tool answers to one model turn share the user turn that
// follows it; the media ride on the last user turn.
- (nullable NSArray *)contentsForRequest:(NFKInferenceRequest *)request
							 attachments:(NFKRemoteAttachments *)attachments
									body:(NSMutableDictionary *)body
{
	NSArray *messages = request.messages;
	if (messages.count == 0) {
		if (request.prompt.length == 0 && attachments.isEmpty) {
			return nil;
		}
		messages = @[ @{ @"role": @"user", @"content": request.prompt ?: @"" } ];
	}
	NSMutableArray<NSDictionary *> *contents = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *answers = [NSMutableArray array];
	for (NSDictionary *message in messages) {
		if (![message isKindOfClass:NSDictionary.class]) {
			continue;
		}
		if ([message[@"role"] isEqual:@"system"]) {
			NSString *system = [self textOfMessageContent:message[@"content"]];
			if (system.length > 0) {
				body[@"systemInstruction"] = @{ @"parts": @[ @{ @"text": system } ] };
			}
			continue;
		}
		if (NFKMessageIsToolResult(message)) {
			[answers addObject:[self functionResponsePartFor:message conversation:messages]];
			continue;
		}
		BOOL fromModel = [message[@"role"] isEqual:@"assistant"];
		NSArray *parts = fromModel ? [self modelPartsFor:message] : [self userPartsFor:message];
		if (answers.count > 0) {
			if (!fromModel) {
				[answers addObjectsFromArray:parts];
				[contents addObject:@{ @"role": @"user", @"parts": [answers copy] }];
				[answers removeAllObjects];
				continue;
			}
			[contents addObject:@{ @"role": @"user", @"parts": [answers copy] }];
			[answers removeAllObjects];
		}
		if (parts.count > 0) {
			[contents addObject:@{ @"role": fromModel ? @"model" : @"user", @"parts": parts }];
		}
	}
	if (answers.count > 0) {
		[contents addObject:@{ @"role": @"user", @"parts": [answers copy] }];
	}
	NSArray *media = [self mediaPartsFor:attachments];
	if (media.count > 0) {
		NSInteger last = -1;
		for (NSInteger index = (NSInteger)contents.count - 1; index >= 0; index--) {
			if ([contents[index][@"role"] isEqual:@"user"]) {
				last = index;
				break;
			}
		}
		if (last < 0) {
			[contents addObject:@{ @"role": @"user", @"parts": media }];
		} else {
			contents[last] = @{ @"role": @"user", @"parts": [contents[last][@"parts"] arrayByAddingObjectsFromArray:media] };
		}
	}
	return contents.count > 0 ? contents : nil;
}

- (NSString *)textOfMessageContent:(id)content
{
	if ([content isKindOfClass:NSString.class]) {
		return content;
	}
	NSMutableArray<NSString *> *texts = [NSMutableArray array];
	for (NSDictionary *item in NFKGenerateContentValue(content, NSArray.class) ?: @[]) {
		NSString *text = [item isKindOfClass:NSDictionary.class] ? NFKGenerateContentValue(item[@"text"], NSString.class) : nil;
		if (text != nil) {
			[texts addObject:text];
		}
	}
	return [texts componentsJoinedByString:@"\n"];
}

- (NSArray *)userPartsFor:(NSDictionary *)message
{
	NSString *text = [self textOfMessageContent:message[@"content"]];
	return text.length > 0 ? @[ @{ @"text": text } ] : @[];
}

// An assistant turn this endpoint wrote replays its own parts, whose thought signatures the API
// checks; any other is rebuilt from its text and calls.
- (NSArray *)modelPartsFor:(NSDictionary *)message
{
	id recorded = NFKMessageProviderTurn(message, NFKProviderTurnFormatGeminiGenerateContent, self.endpointURL.absoluteString);
	if ([recorded isKindOfClass:NSArray.class]) {
		return recorded;
	}
	NSMutableArray *parts = [NSMutableArray array];
	NSString *text = [self textOfMessageContent:message[@"content"]];
	if (text.length > 0) {
		[parts addObject:@{ @"text": text }];
	}
	for (NSDictionary *call in [NFKToolMessages toolCallsOfAssistantMessage:message]) {
		NSMutableDictionary *functionCall = [NSMutableDictionary dictionaryWithObject:call[@"name"] ?: @"" forKey:@"name"];
		functionCall[@"args"] = NFKGenerateContentValue(call[@"arguments"], NSDictionary.class) ?: @{};
		NSString *identifier = NFKGenerateContentValue(call[@"id"], NSString.class);
		if (identifier.length > 0 && ![identifier hasPrefix:NFKGenerateContentGeneratedCallPrefix]) {
			functionCall[@"id"] = identifier;
		}
		[parts addObject:@{ @"functionCall": functionCall }];
	}
	return parts;
}

// The answer's text is the response's output, or its error when the tool failed; images ride as the
// response's own parts.
- (NSDictionary *)functionResponsePartFor:(NSDictionary *)message conversation:(NSArray *)conversation
{
	NSString *text = NFKToolMessageText(message);
	NSMutableDictionary *response = [NSMutableDictionary dictionary];
	response[@"name"] = NFKToolMessageNameIn(message, conversation) ?: @"tool";
	response[@"response"] = NFKToolMessageIsError(message) ? @{ @"error": text } : @{ @"output": text };
	NSString *identifier = NFKToolMessageCallIdentifier(message);
	if (identifier.length > 0 && ![identifier hasPrefix:NFKGenerateContentGeneratedCallPrefix]) {
		response[@"id"] = identifier;
	}
	NSArray<NSData *> *images = NFKToolMessageImagePNGs(message);
	if (images.count > 0) {
		NSMutableArray *parts = [NSMutableArray array];
		for (NSData *png in images) {
			[parts addObject:@{ @"inlineData": @{ @"mimeType": @"image/png", @"data": [png base64EncodedStringWithOptions:0] } }];
		}
		response[@"parts"] = parts;
	}
	return @{ @"functionResponse": response };
}

- (NSArray<NSDictionary *> *)mediaPartsFor:(NFKRemoteAttachments *)attachments
{
	NSMutableArray<NSDictionary *> *parts = [NSMutableArray array];
	for (NSData *png in attachments.imagePNGs) {
		[parts addObject:[self inlinePartWithMIMEType:@"image/png" data:png]];
	}
	if (attachments.audioData != nil) {
		NSString *format = [attachments.audioFormat isEqualToString:@"mp3"] ? @"mpeg" : attachments.audioFormat ?: @"wav";
		[parts addObject:[self inlinePartWithMIMEType:[@"audio/" stringByAppendingString:format] data:attachments.audioData]];
	}
	if (attachments.videoData != nil) {
		NSString *format = [attachments.videoFormat isEqualToString:@"mov"] ? @"quicktime" : attachments.videoFormat ?: @"mp4";
		[parts addObject:[self inlinePartWithMIMEType:[@"video/" stringByAppendingString:format] data:attachments.videoData]];
	}
	for (NSDictionary *document in attachments.documents) {
		NFKRemoteFile *file = document[@"fileReference"];
		if (file != nil) {
			[parts addObject:@{ @"fileData": @{ @"mimeType": file.mimeType ?: @"application/pdf",
												 @"fileUri": file.uri.absoluteString ?: file.identifier } }];
			continue;
		}
		if ([document[@"mediaType"] isEqual:@"text/plain"]) {
			NSString *text = [[NSString alloc] initWithData:document[@"data"] encoding:NSUTF8StringEncoding] ?: @"";
			[parts addObject:@{ @"text": [NSString stringWithFormat:@"%@:\n%@", document[@"filename"], text] }];
			continue;
		}
		[parts addObject:[self inlinePartWithMIMEType:@"application/pdf" data:document[@"data"]]];
	}
	return parts;
}

- (NSDictionary *)inlinePartWithMIMEType:(NSString *)mimeType data:(NSData *)data
{
	return @{ @"inlineData": @{ @"mimeType": mimeType, @"data": [data base64EncodedStringWithOptions:0] } };
}

- (void)addConfigurationForRequest:(NFKInferenceRequest *)request to:(NSMutableDictionary *)body
{
	NSDictionary *parameters = request.parameters;
	NSMutableDictionary *generation = [NSMutableDictionary dictionary];
	generation[@"maxOutputTokens"] = parameters[NFKParameterMaxTokens];
	generation[@"temperature"] = parameters[NFKParameterTemperature];
	generation[@"topP"] = parameters[NFKParameterTopP];
	generation[@"topK"] = parameters[NFKParameterTopK];
	generation[@"seed"] = parameters[NFKParameterSeed];
	generation[@"stopSequences"] = parameters[NFKParameterStopSequences];
	generation[@"thinkingConfig"] = [self thinkingConfigurationForEffort:parameters[NFKParameterReasoningEffort]];
	NSDictionary *schema = NFKGenerateContentValue(parameters[NFKParameterJSONSchema], NSDictionary.class);
	if (schema != nil) {
		generation[@"responseFormat"] = @{ @"text": @{ @"mimeType": @"APPLICATION_JSON", @"schema": schema } };
	}
	NSSet<NSString *> *mapped = NFKGenerateContentMappedParameters();
	NSSet<NSString *> *topLevel = NFKGenerateContentTopLevelParameters();
	for (NSString *key in parameters) {
		if ([topLevel containsObject:key]) {
			body[key] = parameters[key];
		} else if (![mapped containsObject:key]) {
			generation[key] = parameters[key];
		}
	}
	if (generation.count > 0) {
		body[@"generationConfig"] = generation;
	}
	NSArray *tools = [self toolsFor:NFKGenerateContentValue(parameters[NFKParameterTools], NSArray.class)];
	if (tools.count > 0) {
		body[@"tools"] = tools;
	}
}

// A named effort is a thinking level and a number a budget in tokens; summaries come back either way.
- (nullable NSDictionary *)thinkingConfigurationForEffort:(nullable id)effort
{
	NSMutableDictionary *configuration = [NSMutableDictionary dictionaryWithObject:@YES forKey:@"includeThoughts"];
	NSCharacterSet *nonDigits = NSCharacterSet.decimalDigitCharacterSet.invertedSet;
	if ([effort isKindOfClass:NSNumber.class]) {
		configuration[@"thinkingBudget"] = effort;
	} else if ([effort isKindOfClass:NSString.class] && [effort length] > 0 && [effort rangeOfCharacterFromSet:nonDigits].location == NSNotFound) {
		configuration[@"thinkingBudget"] = @([effort integerValue]);
	} else if ([effort isKindOfClass:NSString.class] && [effort length] > 0) {
		NSDictionary<NSString *, NSString *> *levels = @{ NFKReasoningEffortLight: @"LOW", NFKReasoningEffortModerate: @"MEDIUM",
														  NFKReasoningEffortDeep: @"HIGH" };
		configuration[@"thinkingLevel"] = levels[effort] ?: [effort uppercaseString];
	} else {
		return nil;
	}
	return configuration;
}

- (NSArray *)toolsFor:(nullable NSArray *)tools
{
	NSMutableArray *declarations = [NSMutableArray array];
	NSMutableArray *wire = [NSMutableArray array];
	for (NSDictionary *tool in tools ?: @[]) {
		if (![tool isKindOfClass:NSDictionary.class]) {
			continue;
		}
		if (tool[@"name"] == nil) {
			[wire addObject:tool];
			continue;
		}
		NSMutableDictionary *declaration = [NSMutableDictionary dictionaryWithObject:tool[@"name"] forKey:@"name"];
		declaration[@"description"] = tool[@"description"] ?: @"";
		declaration[@"parametersJsonSchema"] = tool[@"parameters"] ?: @{ @"type": @"object", @"properties": @{} };
		[declarations addObject:declaration];
	}
	if (declarations.count > 0) {
		[wire insertObject:@{ @"functionDeclarations": declarations } atIndex:0];
	}
	return wire;
}

#pragma mark Response

- (nullable NFKInferenceResult *)resultFromReply:(NSDictionary *)reply request:(NFKInferenceRequest *)request error:(NSError * _Nullable *)outError
{
	NSString *blockReason = NFKGenerateContentValue(NFKGenerateContentValue(reply[@"promptFeedback"], NSDictionary.class)[@"blockReason"], NSString.class);
	if (blockReason.length > 0) {
		return [self fail:outError code:kNFKError_InferenceRefused reason:[NSString stringWithFormat:@"the prompt was blocked (%@)", blockReason]];
	}
	NSDictionary *candidate = NFKGenerateContentValue([NFKGenerateContentValue(reply[@"candidates"], NSArray.class) firstObject], NSDictionary.class) ?: @{};
	NSArray *parts = NFKGenerateContentValue(NFKGenerateContentValue(candidate[@"content"], NSDictionary.class)[@"parts"], NSArray.class) ?: @[];
	NSMutableArray<NSString *> *thoughts = [NSMutableArray array];
	NSMutableArray *images = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *toolCalls = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *serverTools = [NSMutableArray array];
	for (NSDictionary *part in parts) {
		if (![part isKindOfClass:NSDictionary.class]) {
			continue;
		}
		NSString *partText = NFKGenerateContentValue(part[@"text"], NSString.class);
		if (partText != nil && [part[@"thought"] boolValue]) {
			if (partText.length > 0) {
				[thoughts addObject:partText];
			}
		} else if ([part[@"functionCall"] isKindOfClass:NSDictionary.class]) {
			[toolCalls addObject:[self toolCallFor:part[@"functionCall"] index:toolCalls.count]];
		} else if ([part[@"inlineData"] isKindOfClass:NSDictionary.class]) {
			[self addImageOf:part[@"inlineData"] to:images];
		} else if (part[@"executableCode"] != nil || part[@"codeExecutionResult"] != nil) {
			BOOL code = part[@"executableCode"] != nil;
			[serverTools addObject:@{ @"name": @"code_execution", @"raw": part,
									  code ? @"input" : @"output": code ? part[@"executableCode"] : part[@"codeExecutionResult"] }];
		}
	}
	NSString *text = [self textOfParts:parts];
	NSString *finishReason = NFKGenerateContentValue(candidate[@"finishReason"], NSString.class);
	if (text.length == 0 && toolCalls.count == 0 && images.count == 0 && [NFKGenerateContentRefusingFinishReasons() containsObject:finishReason ?: @""]) {
		NSString *message = NFKGenerateContentValue(candidate[@"finishMessage"], NSString.class);
		NSString *reason = message.length > 0 ? [NSString stringWithFormat:@"the reply was withheld (%@): %@", finishReason, message]
											  : [NSString stringWithFormat:@"the reply was withheld (%@)", finishReason];
		return [self fail:outError code:kNFKError_InferenceRefused reason:reason];
	}
	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionaryWithObject:reply forKey:NFKRemoteBackendRawKey];
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
	if (toolCalls.count > 0) {
		outputs[NFKOutputToolCalls] = toolCalls;
	}
	if (serverTools.count > 0) {
		outputs[NFKOutputServerToolResults] = serverTools;
	}
	NSArray *citations = [self citationsOfCandidate:candidate text:text];
	if (citations.count > 0) {
		outputs[NFKOutputCitations] = citations;
	}
	NSDictionary *usage = NFKGenerateContentValue(reply[@"usageMetadata"], NSDictionary.class);
	NSNumber *candidates = NFKGenerateContentValue(usage[@"candidatesTokenCount"], NSNumber.class);
	NSNumber *thinking = NFKGenerateContentValue(usage[@"thoughtsTokenCount"], NSNumber.class);
	NSNumber *output = candidates != nil || thinking != nil ? @(candidates.integerValue + thinking.integerValue) : nil;
	NSDictionary *counts = NFKRemoteUsage(NFKGenerateContentValue(usage[@"promptTokenCount"], NSNumber.class),
										  NFKGenerateContentValue(usage[@"cachedContentTokenCount"], NSNumber.class), output, thinking);
	if (counts != nil) {
		outputs[NFKOutputUsage] = counts;
	}
	outputs[NFKOutputAssistantMessage] = NFKAssistantMessage(text, toolCalls, NFKProviderTurnFormatGeminiGenerateContent,
															 self.endpointURL.absoluteString, parts);
	return [NFKInferenceResult resultWithOutputs:outputs];
}

- (NSString *)textOfParts:(NSArray *)parts
{
	NSMutableString *text = [NSMutableString string];
	for (NSDictionary *part in parts) {
		NSString *partText = [part isKindOfClass:NSDictionary.class] ? NFKGenerateContentValue(part[@"text"], NSString.class) : nil;
		if (partText != nil && ![part[@"thought"] boolValue]) {
			[text appendString:partText];
		}
	}
	return text;
}

- (NSDictionary *)toolCallFor:(NSDictionary *)call index:(NSUInteger)index
{
	NSDictionary *arguments = NFKGenerateContentValue(call[@"args"], NSDictionary.class) ?: @{};
	NSData *json = [NSJSONSerialization dataWithJSONObject:arguments options:0 error:NULL];
	NSString *identifier = NFKGenerateContentValue(call[@"id"], NSString.class);
	if (identifier.length == 0) {
		identifier = [NSString stringWithFormat:@"%@%lu", NFKGenerateContentGeneratedCallPrefix, (unsigned long)index];
	}
	return @{ @"id": identifier, @"name": NFKGenerateContentValue(call[@"name"], NSString.class) ?: @"", @"arguments": arguments,
			  @"argumentsJSON": json != nil ? [[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding] : @"{}" };
}

- (void)addImageOf:(NSDictionary *)inlineData to:(NSMutableArray *)images
{
	NSString *mimeType = NFKGenerateContentValue(inlineData[@"mimeType"], NSString.class);
	NSString *encoded = NFKGenerateContentValue(inlineData[@"data"], NSString.class);
	if (![mimeType hasPrefix:@"image/"] || encoded == nil) {
		return;
	}
	NSData *bytes = [[NSData alloc] initWithBase64EncodedString:encoded options:NSDataBase64DecodingIgnoreUnknownCharacters];
	CVPixelBufferRef pixelBuffer = bytes != nil ? [NFKImageCoding pixelBufferWithImageData:bytes] : NULL;
	if (pixelBuffer != NULL) {
		[images addObject:(__bridge id)pixelBuffer];
		CVPixelBufferRelease(pixelBuffer);
	}
}

// Grounding supports pair a span of the reply with the search results behind it; recitation
// sources name where a span came from, measured in bytes.
- (NSArray<NSDictionary *> *)citationsOfCandidate:(NSDictionary *)candidate text:(NSString *)text
{
	NSMutableArray<NSDictionary *> *citations = [NSMutableArray array];
	NSDictionary *grounding = NFKGenerateContentValue(candidate[@"groundingMetadata"], NSDictionary.class);
	NSArray *chunks = NFKGenerateContentValue(grounding[@"groundingChunks"], NSArray.class) ?: @[];
	NSArray *supports = NFKGenerateContentValue(grounding[@"groundingSupports"], NSArray.class) ?: @[];
	for (NSDictionary *support in supports) {
		if (![support isKindOfClass:NSDictionary.class]) {
			continue;
		}
		NSDictionary *segment = NFKGenerateContentValue(support[@"segment"], NSDictionary.class);
		for (NSNumber *index in NFKGenerateContentValue(support[@"groundingChunkIndices"], NSArray.class) ?: @[]) {
			NSDictionary *chunk = [index isKindOfClass:NSNumber.class] && index.unsignedIntegerValue < chunks.count ? chunks[index.unsignedIntegerValue] : nil;
			NSDictionary *web = NFKGenerateContentValue(NFKGenerateContentValue(chunk, NSDictionary.class)[@"web"], NSDictionary.class);
			NSMutableDictionary *citation = [NSMutableDictionary dictionaryWithObject:support forKey:@"raw"];
			citation[@"text"] = NFKGenerateContentValue(segment[@"text"], NSString.class) ?: @"";
			citation[@"url"] = web[@"uri"];
			citation[@"title"] = web[@"title"];
			citation[@"start"] = segment[@"startIndex"];
			citation[@"end"] = segment[@"endIndex"];
			[citations addObject:citation];
		}
	}
	NSDictionary *metadata = NFKGenerateContentValue(candidate[@"citationMetadata"], NSDictionary.class);
	for (NSDictionary *source in NFKGenerateContentValue(metadata[@"citationSources"], NSArray.class) ?: @[]) {
		if (![source isKindOfClass:NSDictionary.class]) {
			continue;
		}
		NSMutableDictionary *citation = [NSMutableDictionary dictionaryWithObject:source forKey:@"raw"];
		citation[@"text"] = NFKGenerateContentTextInBytes(text, source[@"startIndex"], source[@"endIndex"]) ?: @"";
		citation[@"url"] = source[@"uri"];
		citation[@"start"] = source[@"startIndex"];
		citation[@"end"] = source[@"endIndex"];
		[citations addObject:citation];
	}
	return citations;
}

#pragma mark Streaming

// Each chunk is a partial reply. Text continues the previous text part of the same kind until that
// part carries a signature; a call, an image, or a signed part starts a new one. Returns YES when
// the visible text grew.
- (BOOL)applyChunk:(NSDictionary *)chunk toState:(NFKGenerateContentStreamState *)state
{
	NSDictionary *error = NFKGenerateContentValue(chunk[@"error"], NSDictionary.class);
	if (error != nil) {
		state.failure = NFKGenerateContentValue(error[@"message"], NSString.class) ?: @"the stream reported an error";
		return NO;
	}
	for (NSString *key in @[ @"usageMetadata", @"promptFeedback", @"modelVersion", @"responseId", @"continuationToken" ]) {
		if (chunk[key] != nil) {
			state.reply[key] = chunk[key];
		}
	}
	NSDictionary *candidate = NFKGenerateContentValue([NFKGenerateContentValue(chunk[@"candidates"], NSArray.class) firstObject], NSDictionary.class);
	for (NSString *key in @[ @"finishReason", @"finishMessage", @"groundingMetadata", @"citationMetadata" ]) {
		if (candidate[key] != nil) {
			state.candidate[key] = candidate[key];
		}
	}
	BOOL grew = NO;
	for (NSDictionary *part in NFKGenerateContentValue(NFKGenerateContentValue(candidate[@"content"], NSDictionary.class)[@"parts"], NSArray.class) ?: @[]) {
		if (![part isKindOfClass:NSDictionary.class]) {
			continue;
		}
		NSString *partText = NFKGenerateContentValue(part[@"text"], NSString.class);
		NSMutableDictionary *previous = state.parts.lastObject;
		BOOL continues = partText != nil && [previous[@"text"] isKindOfClass:NSString.class] && previous[@"thoughtSignature"] == nil &&
						 [previous[@"thought"] boolValue] == [part[@"thought"] boolValue];
		if (continues) {
			previous[@"text"] = [previous[@"text"] stringByAppendingString:partText];
			if (part[@"thoughtSignature"] != nil) {
				previous[@"thoughtSignature"] = part[@"thoughtSignature"];
			}
		} else {
			[state.parts addObject:[part mutableCopy]];
		}
		grew = grew || (partText.length > 0 && ![part[@"thought"] boolValue]);
	}
	return grew;
}

- (NSDictionary *)replyOfState:(NFKGenerateContentStreamState *)state
{
	NSMutableDictionary *candidate = [state.candidate mutableCopy];
	candidate[@"content"] = @{ @"role": @"model", @"parts": [state.parts copy] };
	NSMutableDictionary *reply = [state.reply mutableCopy];
	reply[@"candidates"] = @[ candidate ];
	return reply;
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
