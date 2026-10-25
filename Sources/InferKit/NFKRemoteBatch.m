//
//  NFKRemoteBatch.m
//  InferKit
//

#import <InferKit/NFKRemoteBatch.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKRemoteFileStore.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKErrors.h>
#import "NFKRemoteTransportPrivate.h"
#import "NFKRemoteBatchPrivate.h"

static NSString * _Nullable NFKBatchString(id value)
{
	return [value isKindOfClass:NSString.class] && [value length] > 0 ? value : nil;
}

static NSDictionary * _Nullable NFKBatchDictionary(id value)
{
	return [value isKindOfClass:NSDictionary.class] ? value : nil;
}

/*! A count a service writes as a number or, on Gemini, as an int64 string. */
static NSInteger NFKBatchCount(id value)
{
	return [value isKindOfClass:NSNumber.class] || [value isKindOfClass:NSString.class] ? [value integerValue] : 0;
}

/*! A time a service writes as Unix seconds or as RFC 3339 text. */
static NSDate * _Nullable NFKBatchDate(id value)
{
	if ([value isKindOfClass:NSNumber.class]) {
		return [value doubleValue] > 0 ? [NSDate dateWithTimeIntervalSince1970:[value doubleValue]] : nil;
	}
	if (![value isKindOfClass:NSString.class]) {
		return nil;
	}
	static NSISO8601DateFormatter *plain;
	static NSISO8601DateFormatter *fractional;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		plain = [[NSISO8601DateFormatter alloc] init];
		fractional = [[NSISO8601DateFormatter alloc] init];
		fractional.formatOptions = NSISO8601DateFormatWithInternetDateTime | NSISO8601DateFormatWithFractionalSeconds;
	});
	return [plain dateFromString:value] ?: [fractional dateFromString:value];
}

/*! Each service's status spellings in the shared vocabulary. */
static NSDictionary<NSString *, NSNumber *> *NFKBatchStatuses(void)
{
	static NSDictionary<NSString *, NSNumber *> *statuses;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		statuses = @{ @"validating": @(NFKRemoteBatchStatusQueued), @"queued": @(NFKRemoteBatchStatusQueued),
					  @"pending": @(NFKRemoteBatchStatusQueued), @"in_progress": @(NFKRemoteBatchStatusInProgress),
					  @"running": @(NFKRemoteBatchStatusInProgress), @"finalizing": @(NFKRemoteBatchStatusFinalizing),
					  @"completed": @(NFKRemoteBatchStatusCompleted), @"success": @(NFKRemoteBatchStatusCompleted),
					  @"succeeded": @(NFKRemoteBatchStatusCompleted), @"failed": @(NFKRemoteBatchStatusFailed),
					  @"expired": @(NFKRemoteBatchStatusExpired), @"timeout_exceeded": @(NFKRemoteBatchStatusExpired),
					  @"cancelling": @(NFKRemoteBatchStatusCancelling), @"canceling": @(NFKRemoteBatchStatusCancelling),
					  @"cancellation_requested": @(NFKRemoteBatchStatusCancelling),
					  @"cancelled": @(NFKRemoteBatchStatusCancelled), @"canceled": @(NFKRemoteBatchStatusCancelled) };
	});
	return statuses;
}

/*! A status as the shared vocabulary reads it: lower case, without Gemini's BATCH_STATE_ or JOB_STATE_. */
static NFKRemoteBatchStatus NFKBatchStatusNamed(NSString * _Nullable name)
{
	NSString *key = name.lowercaseString;
	for (NSString *prefix in @[ @"batch_state_", @"job_state_" ]) {
		if ([key hasPrefix:prefix]) {
			key = [key substringFromIndex:prefix.length];
		}
	}
	NSNumber *status = key != nil ? NFKBatchStatuses()[key] : nil;
	return status != nil ? (NFKRemoteBatchStatus)status.integerValue : NFKRemoteBatchStatusUnknown;
}

static NSString * const NFKBatchRouteGenerateContent = @"generateContent";
static NSString * const NFKBatchRouteEmbedContent = @"embedContent";

/*! A request URL's path ending, mapped to the route a batch names: the provider's base differs
	(Groq's /openai/v1, Perplexity's Agent API at /agent), the route a batch reads does not. */
static NSDictionary<NSString *, NSString *> *NFKBatchRouteEndings(void)
{
	return @{ @"chat/completions": @"/v1/chat/completions", @"responses": @"/v1/responses", @"agent": @"/v1/responses",
			  @"messages": @"/v1/messages", @"embeddings": @"/v1/embeddings", @"completions": @"/v1/completions",
			  @"fim/completions": @"/v1/fim/completions", @"ocr": @"/v1/ocr", @"moderations": @"/v1/moderations",
			  @"chat/moderations": @"/v1/chat/moderations", @"classifications": @"/v1/classifications",
			  @"chat/classifications": @"/v1/chat/classifications", @"audio/transcriptions": @"/v1/audio/transcriptions",
			  @"audio/translations": @"/v1/audio/translations", @"images/generations": @"/v1/images/generations",
			  @"images/edits": @"/v1/images/edits", @"videos/generations": @"/v1/videos/generations",
			  @"videos/edits": @"/v1/videos/edits", @"videos/extensions": @"/v1/videos/extensions",
			  @"conversations": @"/v1/conversations" };
}

/*! The key xAI names each batched request's body under, by route. */
static NSDictionary<NSString *, NSString *> *NFKBatchXAIRequestKeys(void)
{
	return @{ @"/v1/chat/completions": @"chat_get_completion", @"/v1/responses": @"responses",
			  @"/v1/images/generations": @"image_generation", @"/v1/images/edits": @"image_edit",
			  @"/v1/videos/generations": @"video_generation", @"/v1/videos/edits": @"video_generation",
			  @"/v1/videos/extensions": @"video_extension" };
}

/*! The routes each style batches. */
static NSDictionary<NSNumber *, NSSet<NSString *> *> *NFKBatchRoutesByStyle(void)
{
	static NSDictionary<NSNumber *, NSSet<NSString *> *> *routes;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		routes = @{
			@(NFKRemoteBatchAPIStyleOpenAI): [NSSet setWithArray:@[ @"/v1/chat/completions", @"/v1/responses", @"/v1/embeddings",
																   @"/v1/completions", @"/v1/moderations", @"/v1/audio/transcriptions",
																   @"/v1/audio/translations", @"/v1/images/generations", @"/v1/images/edits" ]],
			@(NFKRemoteBatchAPIStyleTogether): [NSSet setWithArray:@[ @"/v1/chat/completions" ]],
			@(NFKRemoteBatchAPIStyleAnthropic): [NSSet setWithArray:@[ @"/v1/messages" ]],
			@(NFKRemoteBatchAPIStyleGemini): [NSSet setWithArray:@[ NFKBatchRouteGenerateContent, NFKBatchRouteEmbedContent ]],
			@(NFKRemoteBatchAPIStyleMistral): [NSSet setWithArray:@[ @"/v1/chat/completions", @"/v1/embeddings", @"/v1/fim/completions",
																	@"/v1/ocr", @"/v1/moderations", @"/v1/chat/moderations",
																	@"/v1/classifications", @"/v1/chat/classifications",
																	@"/v1/audio/transcriptions", @"/v1/conversations" ]],
			@(NFKRemoteBatchAPIStyleOpenRouter): [NSSet setWithArray:@[ @"/v1/chat/completions", @"/v1/responses", @"/v1/messages",
																	   @"/v1/embeddings" ]],
			@(NFKRemoteBatchAPIStyleXAI): [NSSet setWithArray:NFKBatchXAIRequestKeys().allKeys],
		};
	});
	return routes;
}

#pragma mark - Record

@interface NFKRemoteBatchRecord ()
@property (nonatomic, copy, readwrite) NSString *identifier;
@property (nonatomic, readwrite) NFKRemoteBatchStatus status;
@property (nonatomic, copy, readwrite, nullable) NSString *serviceStatus;
@property (nonatomic, readwrite) NSInteger requestCount;
@property (nonatomic, readwrite) NSInteger succeededCount;
@property (nonatomic, readwrite) NSInteger failedCount;
@property (nonatomic, copy, readwrite, nullable) NSDate *createdAt;
@property (nonatomic, copy, readwrite, nullable) NSDate *endedAt;
@property (nonatomic, copy, readwrite, nullable) NSDate *expiresAt;
@property (nonatomic, copy, readwrite) NSDictionary<NSString *, id> *raw;
@end

@implementation NFKRemoteBatchRecord

+ (nullable instancetype)recordWithObject:(NSDictionary<NSString *, id> *)object apiStyle:(NFKRemoteBatchAPIStyle)apiStyle
{
	if (![object isKindOfClass:NSDictionary.class]) {
		return nil;
	}
	NFKRemoteBatchRecord *record = [[self alloc] initPrivate];
	record.raw = object;
	BOOL read = NO;
	switch (apiStyle) {
		case NFKRemoteBatchAPIStyleOpenAI:
		case NFKRemoteBatchAPIStyleOpenRouter:
			read = [record readOpenAIObject:object];
			break;
		case NFKRemoteBatchAPIStyleTogether:
			read = [record readTogetherObject:NFKBatchDictionary(object[@"job"]) ?: object];
			break;
		case NFKRemoteBatchAPIStyleAnthropic:
			read = [record readAnthropicObject:object];
			break;
		case NFKRemoteBatchAPIStyleGemini:
			read = [record readGeminiObject:object];
			break;
		case NFKRemoteBatchAPIStyleMistral:
			read = [record readMistralObject:object];
			break;
		case NFKRemoteBatchAPIStyleXAI:
			read = [record readXAIObject:object];
			break;
	}
	return read ? record : nil;
}

- (instancetype)initPrivate
{
	return [super init];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;			// immutable
}

- (BOOL)isFinished
{
	switch (self.status) {
		case NFKRemoteBatchStatusCompleted:
		case NFKRemoteBatchStatusFailed:
		case NFKRemoteBatchStatusExpired:
		case NFKRemoteBatchStatusCancelled:
			return YES;
		default:
			return NO;
	}
}

- (BOOL)readOpenAIObject:(NSDictionary *)object
{
	self.identifier = NFKBatchString(object[@"id"]);
	self.serviceStatus = NFKBatchString(object[@"status"]);
	self.status = NFKBatchStatusNamed(self.serviceStatus);
	NSDictionary *counts = NFKBatchDictionary(object[@"request_counts"]);
	self.requestCount = NFKBatchCount(counts[@"total"]);
	self.succeededCount = NFKBatchCount(counts[@"completed"]);
	self.failedCount = NFKBatchCount(counts[@"failed"]);
	self.createdAt = NFKBatchDate(object[@"created_at"]);
	self.expiresAt = NFKBatchDate(object[@"expires_at"]);
	for (NSString *key in @[ @"completed_at", @"finalized_at", @"failed_at", @"expired_at", @"cancelled_at" ]) {
		self.endedAt = self.endedAt ?: NFKBatchDate(object[key]);
	}
	return self.identifier != nil;
}

- (BOOL)readTogetherObject:(NSDictionary *)object
{
	self.identifier = NFKBatchString(object[@"id"]);
	self.serviceStatus = NFKBatchString(object[@"status"]);
	self.status = NFKBatchStatusNamed(self.serviceStatus);
	self.createdAt = NFKBatchDate(object[@"created_at"]);
	self.endedAt = NFKBatchDate(object[@"completed_at"]);
	self.expiresAt = NFKBatchDate(object[@"job_deadline"]);
	return self.identifier != nil;
}

// An ended batch is cancelled when a cancel was asked for, and completed otherwise.
- (BOOL)readAnthropicObject:(NSDictionary *)object
{
	self.identifier = NFKBatchString(object[@"id"]);
	self.serviceStatus = NFKBatchString(object[@"processing_status"]);
	self.status = NFKBatchStatusNamed(self.serviceStatus);
	if ([self.serviceStatus isEqualToString:@"ended"]) {
		self.status = NFKBatchString(object[@"cancel_initiated_at"]) != nil ? NFKRemoteBatchStatusCancelled : NFKRemoteBatchStatusCompleted;
	}
	NSDictionary *counts = NFKBatchDictionary(object[@"request_counts"]);
	NSInteger failed = NFKBatchCount(counts[@"errored"]) + NFKBatchCount(counts[@"canceled"]) + NFKBatchCount(counts[@"expired"]);
	self.succeededCount = NFKBatchCount(counts[@"succeeded"]);
	self.failedCount = failed;
	self.requestCount = self.succeededCount + failed + NFKBatchCount(counts[@"processing"]);
	self.createdAt = NFKBatchDate(object[@"created_at"]);
	self.endedAt = NFKBatchDate(object[@"ended_at"]);
	self.expiresAt = NFKBatchDate(object[@"expires_at"]);
	return self.identifier != nil;
}

// An operation carries the batch under metadata; a listed batch may come as the resource itself.
- (BOOL)readGeminiObject:(NSDictionary *)object
{
	NSDictionary *batch = NFKBatchDictionary(object[@"metadata"]) ?: object;
	self.identifier = NFKBatchString(object[@"name"]) ?: NFKBatchString(batch[@"name"]);
	self.serviceStatus = NFKBatchString(batch[@"state"]);
	self.status = NFKBatchStatusNamed(self.serviceStatus);
	NSDictionary *stats = NFKBatchDictionary(batch[@"batchStats"]);
	self.requestCount = NFKBatchCount(stats[@"requestCount"]);
	self.succeededCount = NFKBatchCount(stats[@"successfulRequestCount"]);
	self.failedCount = NFKBatchCount(stats[@"failedRequestCount"]);
	self.createdAt = NFKBatchDate(batch[@"createTime"]);
	self.endedAt = NFKBatchDate(batch[@"endTime"]);
	return self.identifier != nil;
}

- (BOOL)readMistralObject:(NSDictionary *)object
{
	self.identifier = NFKBatchString(object[@"id"]);
	self.serviceStatus = NFKBatchString(object[@"status"]);
	self.status = NFKBatchStatusNamed(self.serviceStatus);
	self.requestCount = NFKBatchCount(object[@"total_requests"]);
	self.succeededCount = NFKBatchCount(object[@"succeeded_requests"]);
	self.failedCount = NFKBatchCount(object[@"failed_requests"]);
	self.createdAt = NFKBatchDate(object[@"created_at"]);
	self.endedAt = NFKBatchDate(object[@"completed_at"]);
	return self.identifier != nil;
}

// xAI names no status: a batch with requests pending runs, and one without has stopped.
- (BOOL)readXAIObject:(NSDictionary *)object
{
	self.identifier = NFKBatchString(object[@"batch_id"]);
	NSDictionary *state = NFKBatchDictionary(object[@"state"]);
	self.requestCount = NFKBatchCount(state[@"num_requests"]);
	self.succeededCount = NFKBatchCount(state[@"num_success"]);
	self.failedCount = NFKBatchCount(state[@"num_error"]) + NFKBatchCount(state[@"num_cancelled"]);
	BOOL cancelled = object[@"cancel_time"] != nil && object[@"cancel_time"] != NSNull.null;
	if (NFKBatchCount(state[@"num_pending"]) > 0) {
		self.status = cancelled ? NFKRemoteBatchStatusCancelling : NFKRemoteBatchStatusInProgress;
	} else if (self.requestCount == 0) {
		self.status = cancelled ? NFKRemoteBatchStatusCancelled : NFKRemoteBatchStatusQueued;
	} else {
		self.status = cancelled ? NFKRemoteBatchStatusCancelled : NFKRemoteBatchStatusCompleted;
	}
	self.createdAt = NFKBatchDate(object[@"create_time"]);
	self.expiresAt = NFKBatchDate(object[@"expire_time"]);
	return self.identifier != nil;
}

@end

#pragma mark - Result

@interface NFKRemoteBatchResult ()
@property (nonatomic, copy, readwrite) NSString *customIdentifier;
@property (nonatomic, strong, readwrite, nullable) NFKInferenceResult *result;
@property (nonatomic, strong, readwrite, nullable) NSError *error;
@property (nonatomic, copy, readwrite) NSDictionary<NSString *, id> *raw;
@end

@implementation NFKRemoteBatchResult

- (instancetype)initWithIdentifier:(NSString *)identifier result:(nullable NFKInferenceResult *)result
							 error:(nullable NSError *)error raw:(NSDictionary *)raw
{
	self = [super init];
	if (self != nil) {
		_customIdentifier = [identifier copy];
		_result = result;
		_error = error;
		_raw = [raw copy];
	}
	return self;
}

@end

#pragma mark - Batch

@interface NFKRemoteBatch ()
/*! The inline size past which Gemini's requests go as a file: its inline input takes under 20 MB. */
@property (nonatomic, assign) NSUInteger geminiInlineByteLimit;
@end

@implementation NFKRemoteBatch

@synthesize session = _session;

+ (instancetype)batchWithBaseURL:(nullable NSURL *)baseURL apiStyle:(NFKRemoteBatchAPIStyle)apiStyle
{
	NFKRemoteBatch *batch = [[self alloc] init];
	batch.baseURL = baseURL;
	batch.apiStyle = apiStyle;
	return batch;
}

+ (nullable instancetype)batchForProvider:(NFKRemoteProvider *)provider apiKey:(nullable NSString *)apiKey
{
	NSDictionary<NSString *, NSNumber *> *styles = @{
		@"openai": @(NFKRemoteBatchAPIStyleOpenAI), @"groq": @(NFKRemoteBatchAPIStyleOpenAI),
		@"modelstudio": @(NFKRemoteBatchAPIStyleOpenAI), @"perplexity": @(NFKRemoteBatchAPIStyleOpenAI),
		@"together": @(NFKRemoteBatchAPIStyleTogether), @"anthropic": @(NFKRemoteBatchAPIStyleAnthropic),
		@"gemini": @(NFKRemoteBatchAPIStyleGemini), @"mistral": @(NFKRemoteBatchAPIStyleMistral),
		@"openrouter": @(NFKRemoteBatchAPIStyleOpenRouter), @"xai": @(NFKRemoteBatchAPIStyleXAI) };
	NSNumber *style = styles[provider.identifier];
	if (style == nil) {
		return nil;
	}
	// Gemini's batches sit on its native base, beside the OpenAI layer the preset names.
	NSURL *base = provider.baseURL;
	if (style.integerValue == NFKRemoteBatchAPIStyleGemini) {
		NSString *native = [provider.baseURL.URLByDeletingLastPathComponent.absoluteString
							stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@"/"]];
		base = [NSURL URLWithString:native] ?: base;
	}
	NFKRemoteBatch *batch = [self batchWithBaseURL:base apiStyle:style.integerValue];
	batch.apiKey = apiKey;
	batch.fileStore = [NFKRemoteFileStore fileStoreForProvider:provider apiKey:apiKey];
	NSURLSession *pinned = NFKRemotePinnedSessionForProvider(provider);
	if (pinned != nil) {
		batch.session = pinned;
		batch.fileStore.session = pinned;
	}
	return batch;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_timeout = 600.0;
		_pollInterval = 30.0;
		_geminiInlineByteLimit = 19000000;
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

#pragma mark Submitting

- (nullable NFKRemoteBatchRecord *)submitRequests:(NSDictionary<NSString *, NFKInferenceRequest *> *)requests
										  backend:(id<NFKInferenceBackend>)backend
											error:(NSError * _Nullable *)outError
{
	id<NFKRemoteBatchableBackend> batchable = [self batchableBackend:backend error:outError];
	if (batchable == nil) {
		return nil;
	}
	if (requests.count == 0) {
		return [self failWithCode:kNFKError_InferenceMissingInput reason:@"the batch carries no requests" error:outError];
	}
	NSMutableArray<NSDictionary *> *bodies = [NSMutableArray array];
	NSString *route = nil;
	NSMutableOrderedSet<NSString *> *betas = [NSMutableOrderedSet orderedSet];
	NSArray<NSString *> *identifiers = [requests.allKeys sortedArrayUsingSelector:@selector(compare:)];
	for (NSString *identifier in identifiers) {
		NSURLRequest *urlRequest = [batchable batchURLRequestForRequest:requests[identifier] error:outError];
		NSMutableDictionary *body = urlRequest.HTTPBody != nil
			? [[NSJSONSerialization JSONObjectWithData:urlRequest.HTTPBody options:0 error:NULL] mutableCopy] : nil;
		if (![body isKindOfClass:NSDictionary.class]) {
			return urlRequest == nil ? nil : [self failWithCode:kNFKError_InferenceBackendFailure
														 reason:@"the backend's request has no JSON body" error:outError];
		}
		NSString *lineRoute = [self routeOfURL:urlRequest.URL];
		if (lineRoute == nil || ![self routeIsServed:lineRoute]) {
			return [self failWithCode:kNFKError_InferenceUnsupported
							   reason:[NSString stringWithFormat:@"this service batches no %@ requests", lineRoute ?: urlRequest.URL.path]
								error:outError];
		}
		if (route != nil && ![route isEqualToString:lineRoute]) {
			return [self failWithCode:kNFKError_InferenceUnsupported
							   reason:[NSString stringWithFormat:@"a batch takes one route, and its requests name %@ and %@", route, lineRoute]
								error:outError];
		}
		route = lineRoute;
		[body removeObjectForKey:@"stream"];
		[bodies addObject:body];
		for (NSString *beta in [[urlRequest valueForHTTPHeaderField:@"anthropic-beta"] componentsSeparatedByString:@","] ?: @[]) {
			[betas addObject:beta];
		}
	}
	if (route == nil) {
		return [self failWithCode:kNFKError_InferenceMissingInput reason:@"the batch carries no requests" error:outError];
	}
	switch (self.apiStyle) {
		case NFKRemoteBatchAPIStyleOpenAI:
		case NFKRemoteBatchAPIStyleTogether:
			return [self submitFileOfBodies:bodies identifiers:identifiers route:route error:outError];
		case NFKRemoteBatchAPIStyleAnthropic:
			return [self submitAnthropicBodies:bodies identifiers:identifiers betas:betas.array error:outError];
		case NFKRemoteBatchAPIStyleGemini:
			return [self submitGeminiBodies:bodies identifiers:identifiers route:route model:batchable.batchModelName error:outError];
		case NFKRemoteBatchAPIStyleMistral:
			return [self submitMistralBodies:bodies identifiers:identifiers route:route
									   model:batchable.batchModelName error:outError];
		case NFKRemoteBatchAPIStyleOpenRouter:
			return [self submitOpenRouterBodies:bodies identifiers:identifiers route:route
										  model:batchable.batchModelName error:outError];
		case NFKRemoteBatchAPIStyleXAI:
			return [self submitXAIBodies:bodies identifiers:identifiers route:route error:outError];
	}
	return nil;
}

// OpenAI's lines name the method and the route; Together's carry the body alone.
- (nullable NFKRemoteBatchRecord *)submitFileOfBodies:(NSArray<NSDictionary *> *)bodies
										  identifiers:(NSArray<NSString *> *)identifiers
												route:(NSString *)route
												error:(NSError * _Nullable *)outError
{
	BOOL together = self.apiStyle == NFKRemoteBatchAPIStyleTogether;
	NSMutableArray<NSDictionary *> *lines = [NSMutableArray array];
	for (NSUInteger index = 0; index < bodies.count; index++) {
		[lines addObject:together ? @{ @"custom_id": identifiers[index], @"body": bodies[index] }
								  : @{ @"custom_id": identifiers[index], @"method": @"POST", @"url": route,
									   @"body": bodies[index] }];
	}
	NFKRemoteFile *file = [self uploadLines:lines purpose:together ? @"batch-api" : @"batch" error:outError];
	if (file == nil) {
		return nil;
	}
	// Groq's batch names chat completions as its endpoint whatever its lines name, audio included.
	NSString *endpoint = [route hasPrefix:@"/v1/audio/"] ? @"/v1/chat/completions" : route;
	NSMutableDictionary *create = [NSMutableDictionary dictionaryWithObjectsAndKeys:file.identifier, @"input_file_id",
								   endpoint, @"endpoint", nil];
	if (!together) {
		create[@"completion_window"] = @"24h";
	}
	return [self recordFromPost:create toPath:@"batches" error:outError];
}

- (nullable NFKRemoteBatchRecord *)submitAnthropicBodies:(NSArray<NSDictionary *> *)bodies
											 identifiers:(NSArray<NSString *> *)identifiers
												   betas:(NSArray<NSString *> *)betas
												   error:(NSError * _Nullable *)outError
{
	NSMutableArray *requests = [NSMutableArray array];
	for (NSUInteger index = 0; index < bodies.count; index++) {
		[requests addObject:@{ @"custom_id": identifiers[index], @"params": bodies[index] }];
	}
	NSMutableURLRequest *request = [self JSONRequestTo:[self URLForPath:@"messages/batches"] method:@"POST" body:@{ @"requests": requests }];
	if (betas.count > 0) {
		[request setValue:[betas componentsJoinedByString:@","] forHTTPHeaderField:@"anthropic-beta"];
	}
	return [self recordForRequest:request error:outError];
}

// Each request carries its identifier as metadata, and the responses come back in input order.
- (nullable NFKRemoteBatchRecord *)submitGeminiBodies:(NSArray<NSDictionary *> *)bodies
										  identifiers:(NSArray<NSString *> *)identifiers
												route:(NSString *)route
												model:(nullable NSString *)model
												error:(NSError * _Nullable *)outError
{
	if (model.length == 0) {
		return [self failWithCode:kNFKError_InferenceNotReady reason:@"a Gemini batch needs the backend's model" error:outError];
	}
	NSString *bare = [model hasPrefix:@"models/"] ? [model substringFromIndex:7] : model;
	NSMutableArray *requests = [NSMutableArray array];
	for (NSUInteger index = 0; index < bodies.count; index++) {
		[requests addObject:@{ @"request": bodies[index], @"metadata": @{ @"key": identifiers[index] } }];
	}
	NSDictionary *inputConfig = @{ @"requests": @{ @"requests": requests } };
	NSData *inlineData = [NSJSONSerialization dataWithJSONObject:requests options:0 error:NULL];
	if (inlineData.length > self.geminiInlineByteLimit) {
		// A file line keys its request at the top, beside it.
		NSMutableArray<NSDictionary *> *lines = [NSMutableArray array];
		for (NSUInteger index = 0; index < bodies.count; index++) {
			[lines addObject:@{ @"key": identifiers[index], @"request": bodies[index] }];
		}
		NFKRemoteFile *file = [self uploadLines:lines purpose:nil mimeType:@"jsonl" error:outError];
		if (file == nil) {
			return nil;
		}
		inputConfig = @{ @"fileName": file.identifier };
	}
	NSDictionary *body = @{ @"batch": @{ @"displayName": @"inferkit-batch", @"inputConfig": inputConfig } };
	NSString *method = [route isEqualToString:NFKBatchRouteEmbedContent] ? @"asyncBatchEmbedContent" : @"batchGenerateContent";
	return [self recordFromPost:body toPath:[NSString stringWithFormat:@"models/%@:%@", bare, method] error:outError];
}

// Up to 10,000 requests go inline; more go as an uploaded file.
- (nullable NFKRemoteBatchRecord *)submitMistralBodies:(NSArray<NSDictionary *> *)bodies
										   identifiers:(NSArray<NSString *> *)identifiers
												 route:(NSString *)route
												 model:(nullable NSString *)model
												 error:(NSError * _Nullable *)outError
{
	NSMutableArray<NSDictionary *> *lines = [NSMutableArray array];
	for (NSUInteger index = 0; index < bodies.count; index++) {
		[lines addObject:@{ @"custom_id": identifiers[index], @"body": bodies[index] }];
	}
	NSMutableDictionary *create = [NSMutableDictionary dictionaryWithObject:route forKey:@"endpoint"];
	create[@"model"] = model;
	if (lines.count <= 10000) {
		create[@"requests"] = lines;
	} else {
		NFKRemoteFile *file = [self uploadLines:lines purpose:@"batch" error:outError];
		if (file == nil) {
			return nil;
		}
		create[@"input_files"] = @[ file.identifier ];
	}
	return [self recordFromPost:create toPath:@"batch/jobs" error:outError];
}

- (nullable NFKRemoteBatchRecord *)submitOpenRouterBodies:(NSArray<NSDictionary *> *)bodies
											  identifiers:(NSArray<NSString *> *)identifiers
													route:(NSString *)route
													model:(nullable NSString *)model
													error:(NSError * _Nullable *)outError
{
	if (model.length == 0) {
		return [self failWithCode:kNFKError_InferenceNotReady reason:@"an OpenRouter batch needs the backend's model" error:outError];
	}
	NSMutableArray<NSDictionary *> *requests = [NSMutableArray array];
	for (NSUInteger index = 0; index < bodies.count; index++) {
		[requests addObject:@{ @"custom_id": identifiers[index], @"body": bodies[index] }];
	}
	return [self recordFromPost:@{ @"endpoint": route, @"model": model, @"requests": requests }
						 toPath:@"batches" error:outError];
}

// The batch is created by name, the requests are added in pieces under the service's 25 MB a call,
// and the record is read once they are in.
- (nullable NFKRemoteBatchRecord *)submitXAIBodies:(NSArray<NSDictionary *> *)bodies
									   identifiers:(NSArray<NSString *> *)identifiers
											 route:(NSString *)route
											 error:(NSError * _Nullable *)outError
{
	NSString *name = [@"inferkit-" stringByAppendingString:NSUUID.UUID.UUIDString.lowercaseString];
	NFKRemoteBatchRecord *created = [self recordFromPost:@{ @"name": name } toPath:@"batches" error:outError];
	if (created == nil) {
		return nil;
	}
	NSString *key = NFKBatchXAIRequestKeys()[route];
	NSString *path = [NSString stringWithFormat:@"batches/%@/requests", [self escapedSegment:created.identifier]];
	const NSUInteger piece = 100;
	for (NSUInteger start = 0; start < bodies.count; start += piece) {
		NSMutableArray *requests = [NSMutableArray array];
		for (NSUInteger index = start; index < MIN(start + piece, bodies.count); index++) {
			[requests addObject:@{ @"batch_request_id": identifiers[index], @"batch_request": @{ key: bodies[index] } }];
		}
		NSURLRequest *add = [self JSONRequestTo:[self URLForPath:path] method:@"POST" body:@{ @"batch_requests": requests }];
		if ([self dataForRequest:add error:outError] == nil) {
			return nil;
		}
	}
	return [self recordOfBatchWithIdentifier:created.identifier error:outError];
}

- (nullable NFKRemoteFile *)uploadLines:(NSArray<NSDictionary *> *)lines purpose:(nullable NSString *)purpose error:(NSError * _Nullable *)outError
{
	return [self uploadLines:lines purpose:purpose mimeType:@"application/jsonl" error:outError];
}

// Gemini's guide stores a batch file under the MIME type jsonl.
- (nullable NFKRemoteFile *)uploadLines:(NSArray<NSDictionary *> *)lines
								purpose:(nullable NSString *)purpose
							   mimeType:(NSString *)mimeType
								  error:(NSError * _Nullable *)outError
{
	if (self.fileStore == nil) {
		return [self failWithCode:kNFKError_InferenceNotReady reason:@"a file-based batch needs a file store" error:outError];
	}
	NSMutableData *jsonl = [NSMutableData data];
	for (NSDictionary *line in lines) {
		NSData *data = [NSJSONSerialization dataWithJSONObject:line options:NSJSONWritingSortedKeys error:outError];
		if (data == nil) {
			return nil;
		}
		[jsonl appendData:data];
		[jsonl appendData:[@"\n" dataUsingEncoding:NSUTF8StringEncoding]];
	}
	NSString *filename = [NSString stringWithFormat:@"inferkit-batch-%@.jsonl", NSUUID.UUID.UUIDString.lowercaseString];
	return [self.fileStore uploadData:jsonl filename:filename mimeType:mimeType purpose:purpose expiresAfter:0 error:outError];
}

#pragma mark Reading

- (nullable NFKRemoteBatchRecord *)recordOfBatchWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError
{
	return [self recordForRequest:[self requestTo:[self URLForPath:[self pathOfBatch:identifier suffix:nil]] method:@"GET"] error:outError];
}

- (nullable NFKRemoteBatchRecord *)recordWhenFinishedOfBatchWithIdentifier:(NSString *)identifier
																   timeout:(NSTimeInterval)timeout
																	 error:(NSError * _Nullable *)outError
{
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout];
	NFKRemoteBatchRecord *record = [self recordOfBatchWithIdentifier:identifier error:outError];
	while (record != nil && !record.isFinished && deadline.timeIntervalSinceNow > 0) {
		[NSThread sleepForTimeInterval:MIN(self.pollInterval, MAX(deadline.timeIntervalSinceNow, 0))];
		record = [self recordOfBatchWithIdentifier:identifier error:outError];
	}
	return record;
}

- (nullable NSArray<NFKRemoteBatchRecord *> *)batchesWithError:(NSError * _Nullable *)outError
{
	NSArray<NSDictionary *> *objects = [self listedObjectsWithError:outError];
	if (objects == nil) {
		return nil;
	}
	NSMutableArray<NFKRemoteBatchRecord *> *records = [NSMutableArray array];
	for (NSDictionary *object in objects) {
		NFKRemoteBatchRecord *record = [NFKRemoteBatchRecord recordWithObject:object apiStyle:self.apiStyle];
		if (record != nil) {
			[records addObject:record];
		}
	}
	return records;
}

/*! Every listed batch across the pages, each service paging its own way. A token seen before ends the
	reading. */
- (nullable NSArray<NSDictionary *> *)listedObjectsWithError:(NSError * _Nullable *)outError
{
	NSMutableArray<NSDictionary *> *objects = [NSMutableArray array];
	NSMutableSet<NSString *> *seen = [NSMutableSet set];
	NSString *token = nil;
	NSInteger page = 0;
	do {
		NSMutableDictionary<NSString *, NSString *> *query = [NSMutableDictionary dictionary];
		NSString *path = self.apiStyle == NFKRemoteBatchAPIStyleAnthropic ? @"messages/batches"
					   : self.apiStyle == NFKRemoteBatchAPIStyleMistral ? @"batch/jobs" : @"batches";
		switch (self.apiStyle) {
			case NFKRemoteBatchAPIStyleAnthropic:
				query[@"limit"] = @"1000";
				query[@"after_id"] = token;
				break;
			case NFKRemoteBatchAPIStyleGemini:
				query[@"pageSize"] = @"1000";
				query[@"pageToken"] = token;
				break;
			case NFKRemoteBatchAPIStyleMistral:
				query[@"listingsize"] = @"100";
				query[@"page"] = [NSString stringWithFormat:@"%ld", (long)page];
				break;
			case NFKRemoteBatchAPIStyleXAI:
				query[@"limit"] = @"100";
				query[@"pagination_token"] = token;
				break;
			case NFKRemoteBatchAPIStyleTogether:
				break;
			case NFKRemoteBatchAPIStyleOpenAI:
			case NFKRemoteBatchAPIStyleOpenRouter:
				query[@"limit"] = @"100";
				// Groq pages by cursor where OpenAI pages by the last id.
				query[[token hasPrefix:@"cursor:"] ? @"cursor" : @"after"] = [token hasPrefix:@"cursor:"] ? [token substringFromIndex:7] : token;
				break;
		}
		NSData *data = [self dataForRequest:[self requestTo:[self URLForPath:path query:query] method:@"GET"] error:outError];
		id listing = data != nil ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
		if (data == nil) {
			return nil;
		}
		NSArray *entries = [listing isKindOfClass:NSArray.class] ? listing
						 : [listing[@"data"] isKindOfClass:NSArray.class] ? listing[@"data"]
						 : [listing[@"operations"] isKindOfClass:NSArray.class] ? listing[@"operations"]
						 : [listing[@"batches"] isKindOfClass:NSArray.class] ? listing[@"batches"] : nil;
		if (entries == nil) {
			return [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the batch list carries no batches" error:outError];
		}
		[objects addObjectsFromArray:entries];
		token = [self nextTokenInPage:[listing isKindOfClass:NSDictionary.class] ? listing : @{} listedSoFar:objects.count];
		page++;
		if (token != nil && [seen containsObject:token]) {
			break;
		}
		if (token != nil) {
			[seen addObject:token];
		}
	} while (token != nil);
	return objects;
}

- (nullable NSString *)nextTokenInPage:(NSDictionary *)page listedSoFar:(NSUInteger)count
{
	switch (self.apiStyle) {
		case NFKRemoteBatchAPIStyleAnthropic:
		case NFKRemoteBatchAPIStyleOpenAI:
		case NFKRemoteBatchAPIStyleOpenRouter: {
			NSString *cursor = NFKBatchString(NFKBatchDictionary(page[@"paging"])[@"next_cursor"]);
			if (cursor != nil) {
				return [@"cursor:" stringByAppendingString:cursor];
			}
			return [page[@"has_more"] boolValue] ? NFKBatchString(page[@"last_id"]) : nil;
		}
		case NFKRemoteBatchAPIStyleGemini:
			return NFKBatchString(page[@"nextPageToken"]);
		case NFKRemoteBatchAPIStyleXAI:
			return NFKBatchString(page[@"pagination_token"]);
		case NFKRemoteBatchAPIStyleMistral:
			return (NSInteger)count < NFKBatchCount(page[@"total"]) && [page[@"data"] count] > 0
				? [NSString stringWithFormat:@"page-%lu", (unsigned long)count] : nil;
		case NFKRemoteBatchAPIStyleTogether:
			return nil;
	}
	return nil;
}

#pragma mark Cancelling and deleting

- (BOOL)cancelBatchWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError
{
	if (self.apiStyle == NFKRemoteBatchAPIStyleOpenRouter) {
		[self failWithCode:kNFKError_InferenceUnsupported reason:@"OpenRouter's batches cannot be cancelled" error:outError];
		return NO;
	}
	BOOL colon = self.apiStyle == NFKRemoteBatchAPIStyleGemini || self.apiStyle == NFKRemoteBatchAPIStyleXAI;
	NSString *path = [self pathOfBatch:identifier suffix:colon ? @":cancel" : @"/cancel"];
	return [self dataForRequest:[self JSONRequestTo:[self URLForPath:path] method:@"POST" body:@{}] error:outError] != nil;
}

- (BOOL)deleteBatchWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError
{
	NSArray<NSNumber *> *deleting = @[ @(NFKRemoteBatchAPIStyleAnthropic), @(NFKRemoteBatchAPIStyleGemini),
									   @(NFKRemoteBatchAPIStyleMistral), @(NFKRemoteBatchAPIStyleOpenRouter) ];
	if (![deleting containsObject:@(self.apiStyle)]) {
		[self failWithCode:kNFKError_InferenceUnsupported reason:@"this service deletes no batch" error:outError];
		return NO;
	}
	return [self dataForRequest:[self requestTo:[self URLForPath:[self pathOfBatch:identifier suffix:nil]] method:@"DELETE"] error:outError] != nil;
}

#pragma mark Results

- (nullable NSDictionary<NSString *, NFKRemoteBatchResult *> *)resultsOfBatchWithIdentifier:(NSString *)identifier
																				   requests:(nullable NSDictionary<NSString *, NFKInferenceRequest *> *)requests
																					backend:(id<NFKInferenceBackend>)backend
																					  error:(NSError * _Nullable *)outError
{
	id<NFKRemoteBatchableBackend> batchable = [self batchableBackend:backend error:outError];
	if (batchable == nil) {
		return nil;
	}
	NSArray<NSDictionary *> *entries = [self resultEntriesOfBatch:identifier requests:requests error:outError];
	if (entries == nil) {
		return nil;
	}
	NSMutableDictionary<NSString *, NFKRemoteBatchResult *> *results = [NSMutableDictionary dictionary];
	for (NSDictionary *entry in entries) {
		NFKRemoteBatchResult *result = [self resultForEntry:entry requests:requests backend:batchable];
		if (result != nil) {
			results[result.customIdentifier] = result;
		}
	}
	return results;
}

/*! The service's per-request entries: an output and an error file, a results stream, or the entries
	the batch record holds. */
- (nullable NSArray<NSDictionary *> *)resultEntriesOfBatch:(NSString *)identifier
												  requests:(nullable NSDictionary<NSString *, NFKInferenceRequest *> *)requests
													 error:(NSError * _Nullable *)outError
{
	switch (self.apiStyle) {
		case NFKRemoteBatchAPIStyleOpenAI:
		case NFKRemoteBatchAPIStyleTogether:
		case NFKRemoteBatchAPIStyleMistral: {
			NFKRemoteBatchRecord *record = [self recordOfBatchWithIdentifier:identifier error:outError];
			if (record == nil) {
				return nil;
			}
			NSDictionary *object = NFKBatchDictionary(record.raw[@"job"]) ?: record.raw;
			BOOL mistral = self.apiStyle == NFKRemoteBatchAPIStyleMistral;
			NSMutableArray<NSDictionary *> *entries = [NSMutableArray array];
			for (NSString *key in mistral ? @[ @"output_file", @"error_file" ] : @[ @"output_file_id", @"error_file_id" ]) {
				NSString *file = NFKBatchString(object[key]);
				NSArray *lines = file != nil ? [self linesOfFile:file error:outError] : @[];
				if (lines == nil) {
					return nil;
				}
				[entries addObjectsFromArray:lines];
			}
			return entries;
		}
		case NFKRemoteBatchAPIStyleOpenRouter: {
			NFKRemoteBatchRecord *record = [self recordOfBatchWithIdentifier:identifier error:outError];
			NSArray *results = [record.raw[@"results"] isKindOfClass:NSArray.class] ? record.raw[@"results"] : @[];
			return record != nil ? results : nil;
		}
		case NFKRemoteBatchAPIStyleAnthropic: {
			NFKRemoteBatchRecord *record = [self recordOfBatchWithIdentifier:identifier error:outError];
			if (record == nil) {
				return nil;
			}
			NSString *link = NFKBatchString(record.raw[@"results_url"]);
			if (link == nil) {
				return @[];
			}
			NSData *data = [self dataForRequest:[self requestTo:[NSURL URLWithString:link] method:@"GET"] error:outError];
			return data != nil ? [self linesInData:data] : nil;
		}
		case NFKRemoteBatchAPIStyleGemini:
			return [self geminiEntriesOfBatch:identifier requests:requests error:outError];
		case NFKRemoteBatchAPIStyleXAI:
			return [self xAIEntriesOfBatch:identifier error:outError];
	}
	return nil;
}

// The responses come inline, or as a file to download from the host's root, under the batch's
// metadata.output (the SDK's reading) or its response (the guide's). Each names its request
// through its metadata; one without names it by position, the requests going in identifier order.
- (nullable NSArray<NSDictionary *> *)geminiEntriesOfBatch:(NSString *)identifier
												  requests:(nullable NSDictionary<NSString *, NFKInferenceRequest *> *)requests
													 error:(NSError * _Nullable *)outError
{
	NFKRemoteBatchRecord *record = [self recordOfBatchWithIdentifier:identifier error:outError];
	if (record == nil) {
		return nil;
	}
	NSDictionary *output = nil;
	for (NSDictionary *candidate in @[ NFKBatchDictionary(NFKBatchDictionary(record.raw[@"metadata"])[@"output"]) ?: @{},
									   NFKBatchDictionary(record.raw[@"response"]) ?: @{},
									   NFKBatchDictionary(NFKBatchDictionary(record.raw[@"response"])[@"output"]) ?: @{} ]) {
		if (candidate[@"inlinedResponses"] != nil || candidate[@"responsesFile"] != nil) {
			output = candidate;
			break;
		}
	}
	NSArray *entries = nil;
	NSDictionary *inlined = NFKBatchDictionary(output[@"inlinedResponses"]);
	if (inlined != nil) {
		entries = [inlined[@"inlinedResponses"] isKindOfClass:NSArray.class] ? inlined[@"inlinedResponses"] : @[];
	} else if (NFKBatchString(output[@"responsesFile"]) != nil) {
		NSURLComponents *components = [NSURLComponents componentsWithURL:self.baseURL resolvingAgainstBaseURL:NO];
		components.path = [NSString stringWithFormat:@"/download/v1beta/%@:download", output[@"responsesFile"]];
		components.queryItems = @[ [NSURLQueryItem queryItemWithName:@"alt" value:@"media"] ];
		NSData *data = components.URL != nil ? [self dataForRequest:[self requestTo:components.URL method:@"GET"] error:outError] : nil;
		if (data == nil) {
			return nil;
		}
		entries = [self linesInData:data];
	} else {
		return @[];
	}
	NSArray<NSString *> *order = [requests.allKeys sortedArrayUsingSelector:@selector(compare:)];
	NSMutableArray<NSDictionary *> *keyed = [NSMutableArray array];
	for (NSUInteger index = 0; index < entries.count; index++) {
		NSDictionary *entry = NFKBatchDictionary(entries[index]);
		NSString *key = NFKBatchString(NFKBatchDictionary(entry[@"metadata"])[@"key"]) ?: NFKBatchString(entry[@"key"])
					  ?: (index < order.count ? order[index] : nil);
		if (entry != nil && key != nil) {
			NSMutableDictionary *named = [entry mutableCopy];
			named[@"key"] = key;
			[keyed addObject:named];
		}
	}
	return keyed;
}

- (nullable NSArray<NSDictionary *> *)xAIEntriesOfBatch:(NSString *)identifier error:(NSError * _Nullable *)outError
{
	NSMutableArray<NSDictionary *> *entries = [NSMutableArray array];
	NSMutableSet<NSString *> *seen = [NSMutableSet set];
	NSString *token = nil;
	do {
		NSMutableDictionary *query = [NSMutableDictionary dictionaryWithObject:@"1000" forKey:@"limit"];
		query[@"pagination_token"] = token;
		NSURL *url = [self URLForPath:[self pathOfBatch:identifier suffix:@"/results"] query:query];
		NSData *data = [self dataForRequest:[self requestTo:url method:@"GET"] error:outError];
		NSDictionary *page = data != nil ? NFKBatchDictionary([NSJSONSerialization JSONObjectWithData:data options:0 error:NULL]) : nil;
		if (page == nil) {
			return data == nil ? nil : [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the results are not a JSON object" error:outError];
		}
		[entries addObjectsFromArray:[page[@"results"] isKindOfClass:NSArray.class] ? page[@"results"] : @[]];
		token = NFKBatchString(page[@"pagination_token"]);
		if (token != nil && [seen containsObject:token]) {
			break;
		}
		if (token != nil) {
			[seen addObject:token];
		}
	} while (token != nil);
	return entries;
}

/*! One entry as a result: its identifier, and the reply read by the backend or the reason it failed. */
- (nullable NFKRemoteBatchResult *)resultForEntry:(NSDictionary *)entry
										 requests:(nullable NSDictionary<NSString *, NFKInferenceRequest *> *)requests
										  backend:(id<NFKRemoteBatchableBackend>)backend
{
	if (![entry isKindOfClass:NSDictionary.class]) {
		return nil;
	}
	NSString *identifier = NFKBatchString(entry[@"custom_id"]) ?: NFKBatchString(entry[@"batch_request_id"]) ?: NFKBatchString(entry[@"key"]);
	if (identifier == nil) {
		return nil;
	}
	NSDictionary *reply = nil;
	NSError *error = [self errorInEntry:entry reply:&reply];
	NFKInferenceResult *result = nil;
	if (error == nil && reply != nil) {
		NFKInferenceRequest *request = requests[identifier] ?: [NFKInferenceRequest requestWithInputs:@{}];
		NSError *readError = nil;
		result = [backend batchResultForReply:reply request:request error:&readError];
		error = result == nil ? readError : nil;
	} else if (error == nil) {
		error = [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the entry carries neither a reply nor an error"];
	}
	return [[NFKRemoteBatchResult alloc] initWithIdentifier:identifier result:result error:error raw:entry];
}

/*! The entry's failure, or nil with the reply it carries. The shapes: OpenAI's {response: {status_code,
	body}, error}, Anthropic's {result: {type, message | error}}, Gemini's {response | error}, and
	xAI's {batch_result: {response: {<route>: body} | error}}. */
- (nullable NSError *)errorInEntry:(NSDictionary *)entry reply:(NSDictionary * _Nullable * _Nonnull)outReply
{
	*outReply = nil;
	NSDictionary *anthropic = NFKBatchDictionary(entry[@"result"]);
	if (anthropic != nil) {
		NSString *type = NFKBatchString(anthropic[@"type"]);
		if ([type isEqualToString:@"succeeded"]) {
			*outReply = NFKBatchDictionary(anthropic[@"message"]);
			return nil;
		}
		NSDictionary *wrapped = NFKBatchDictionary(anthropic[@"error"]);
		NSDictionary *detail = NFKBatchDictionary(wrapped[@"error"]) ?: wrapped;
		return [self errorWithMessage:NFKBatchString(detail[@"message"]) ?: [NSString stringWithFormat:@"the request was %@", type ?: @"not run"]];
	}
	NSDictionary *xAI = NFKBatchDictionary(entry[@"batch_result"]);
	if (xAI != nil) {
		NSDictionary *response = NFKBatchDictionary(xAI[@"response"]);
		for (id value in response.allValues) {
			*outReply = NFKBatchDictionary(value) ?: *outReply;
		}
		id failure = xAI[@"error"];
		return *outReply != nil ? nil : [self errorWithMessage:NFKBatchString(failure) ?: NFKBatchString(NFKBatchDictionary(failure)[@"message"]) ?: @"the request failed"];
	}
	id failure = entry[@"error"];
	if ([failure isKindOfClass:NSDictionary.class] || [failure isKindOfClass:NSString.class]) {
		NSString *message = NFKBatchString(failure) ?: NFKBatchString(failure[@"message"]) ?: NFKBatchString(failure[@"code"]) ?: @"the request failed";
		return [self errorWithMessage:message];
	}
	NSDictionary *response = NFKBatchDictionary(entry[@"response"]);
	if (response[@"status_code"] == nil) {
		*outReply = response;
		return nil;
	}
	NSInteger status = NFKBatchCount(response[@"status_code"]);
	NSDictionary *body = NFKBatchDictionary(response[@"body"]);
	if (status >= 200 && status < 300) {
		*outReply = body;
		return nil;
	}
	NSData *data = body != nil ? [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL] : nil;
	NSHTTPURLResponse *http = [[NSHTTPURLResponse alloc] initWithURL:self.baseURL ?: [NSURL URLWithString:@"about:blank"]
														  statusCode:status HTTPVersion:@"HTTP/1.1" headerFields:nil];
	return [NFKRemoteTransport errorForResponse:http data:data] ?: [self errorWithMessage:@"the request failed"];
}

- (NSError *)errorWithMessage:(NSString *)message
{
	return [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:message];
}

- (nullable NSArray<NSDictionary *> *)linesOfFile:(NSString *)identifier error:(NSError * _Nullable *)outError
{
	if (self.fileStore == nil) {
		return [self failWithCode:kNFKError_InferenceNotReady reason:@"a file-based batch needs a file store" error:outError];
	}
	NSData *data = [self.fileStore contentsOfFileWithIdentifier:identifier error:outError];
	return data != nil ? [self linesInData:data] : nil;
}

- (NSArray<NSDictionary *> *)linesInData:(NSData *)data
{
	NSString *text = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] ?: @"";
	NSMutableArray<NSDictionary *> *lines = [NSMutableArray array];
	for (NSString *line in [text componentsSeparatedByCharactersInSet:NSCharacterSet.newlineCharacterSet]) {
		NSData *lineData = [line dataUsingEncoding:NSUTF8StringEncoding];
		NSDictionary *object = line.length > 0 ? NFKBatchDictionary([NSJSONSerialization JSONObjectWithData:lineData options:0 error:NULL]) : nil;
		if (object != nil) {
			[lines addObject:object];
		}
	}
	return lines;
}

#pragma mark Routes

- (nullable id<NFKRemoteBatchableBackend>)batchableBackend:(id<NFKInferenceBackend>)backend error:(NSError * _Nullable *)outError
{
	if ([(id)backend conformsToProtocol:@protocol(NFKRemoteBatchableBackend)]) {
		return (id<NFKRemoteBatchableBackend>)backend;
	}
	return [self failWithCode:kNFKError_InferenceUnsupported
					   reason:@"a batch runs through NFKRemoteBackend, NFKAnthropicBackend, NFKRemoteResponsesBackend, NFKGeminiGenerateContentBackend, or NFKRemoteEmbeddingBackend"
						error:outError];
}

- (BOOL)routeIsServed:(NSString *)route
{
	return [NFKBatchRoutesByStyle()[@(self.apiStyle)] containsObject:route];
}

/*! The route a request's URL names, as every batch API spells it whatever the provider's own base:
	the longest known path ending, or Gemini's :generateContent and :embedContent methods. */
- (nullable NSString *)routeOfURL:(nullable NSURL *)url
{
	NSString *last = url.lastPathComponent;
	for (NSString *method in @[ NFKBatchRouteGenerateContent, NFKBatchRouteEmbedContent ]) {
		if ([last hasSuffix:[@":" stringByAppendingString:method]]) {
			return method;
		}
	}
	NSString *path = url.path;
	NSString *found = nil;
	for (NSString *ending in NFKBatchRouteEndings()) {
		if ([path hasSuffix:[@"/" stringByAppendingString:ending]] && ending.length > found.length) {
			found = ending;
		}
	}
	return found != nil ? NFKBatchRouteEndings()[found] : nil;
}

- (NSString *)pathOfBatch:(NSString *)identifier suffix:(nullable NSString *)suffix
{
	NSString *path = nil;
	switch (self.apiStyle) {
		case NFKRemoteBatchAPIStyleAnthropic:
			path = [@"messages/batches/" stringByAppendingString:[self escapedSegment:identifier]];
			break;
		case NFKRemoteBatchAPIStyleGemini:
			path = [identifier hasPrefix:@"batches/"] ? identifier : [@"batches/" stringByAppendingString:[self escapedSegment:identifier]];
			break;
		case NFKRemoteBatchAPIStyleMistral:
			path = [@"batch/jobs/" stringByAppendingString:[self escapedSegment:identifier]];
			break;
		default:
			path = [@"batches/" stringByAppendingString:[self escapedSegment:identifier]];
			break;
	}
	return suffix != nil ? [path stringByAppendingString:suffix] : path;
}

- (NSString *)escapedSegment:(NSString *)segment
{
	NSMutableCharacterSet *allowed = [NSCharacterSet.URLPathAllowedCharacterSet mutableCopy];
	[allowed removeCharactersInString:@"/:"];
	return [segment stringByAddingPercentEncodingWithAllowedCharacters:allowed] ?: segment;
}

#pragma mark Requests

- (NSURL *)URLForPath:(NSString *)path
{
	return [self URLForPath:path query:@{}];
}

- (NSURL *)URLForPath:(NSString *)path query:(NSDictionary<NSString *, NSString *> *)query
{
	NSString *base = [self.baseURL.absoluteString stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@"/"]] ?: @"";
	NSURL *url = [NSURL URLWithString:[NSString stringWithFormat:@"%@/%@", base, path]];
	if (query.count == 0 || url == nil) {
		return url ?: self.baseURL;
	}
	NSURLComponents *components = [NSURLComponents componentsWithURL:url resolvingAgainstBaseURL:NO];
	NSMutableArray<NSURLQueryItem *> *items = [NSMutableArray array];
	for (NSString *name in [query.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		[items addObject:[NSURLQueryItem queryItemWithName:name value:query[name]]];
	}
	components.queryItems = items;
	return components.URL ?: url;
}

- (NSMutableURLRequest *)requestTo:(NSURL *)url method:(NSString *)method
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.HTTPMethod = method;
	request.timeoutInterval = self.timeout;
	if (self.apiStyle == NFKRemoteBatchAPIStyleGemini) {
		if (self.apiKey.length > 0) {
			[request setValue:self.apiKey forHTTPHeaderField:@"x-goog-api-key"];
		}
		return request;
	}
	[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey
								   style:self.apiStyle == NFKRemoteBatchAPIStyleAnthropic ? NFKRemoteAPIStyleAnthropicMessages : NFKRemoteAPIStyleOpenAIChat];
	return request;
}

// Sorted keys put OpenRouter's requests after the fields it reads first, which it requires.
- (NSMutableURLRequest *)JSONRequestTo:(NSURL *)url method:(NSString *)method body:(NSDictionary *)body
{
	NSMutableURLRequest *request = [self requestTo:url method:method];
	request.HTTPBody = [NSJSONSerialization dataWithJSONObject:body options:NSJSONWritingSortedKeys error:NULL];
	[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	return request;
}

- (nullable NFKRemoteBatchRecord *)recordFromPost:(NSDictionary *)body toPath:(NSString *)path error:(NSError * _Nullable *)outError
{
	return [self recordForRequest:[self JSONRequestTo:[self URLForPath:path] method:@"POST" body:body] error:outError];
}

- (nullable NFKRemoteBatchRecord *)recordForRequest:(NSURLRequest *)request error:(NSError * _Nullable *)outError
{
	NSData *data = [self dataForRequest:request error:outError];
	if (data == nil) {
		return nil;
	}
	NSDictionary *object = NFKBatchDictionary([NSJSONSerialization JSONObjectWithData:data options:0 error:NULL]);
	NFKRemoteBatchRecord *record = object != nil ? [NFKRemoteBatchRecord recordWithObject:object apiStyle:self.apiStyle] : nil;
	return record ?: [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the answer carries no batch" error:outError];
}

#pragma mark Transport

- (nullable NSData *)dataForRequest:(NSURLRequest *)request error:(NSError * _Nullable *)outError
{
	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:request response:&response error:&sendError];
	NSError *failure = data == nil ? sendError : [NFKRemoteTransport errorForResponse:response data:data];
	if (failure != nil || data == nil) {
		if (outError != NULL) {
			*outError = failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the batch call failed"];
		}
		return nil;
	}
	return data;
}

- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	return [NFKRemoteTransport sendRequest:request session:self.session response:outResponse error:outError];
}

- (nullable id)failWithCode:(NFKInferenceError)code reason:(NSString *)reason error:(NSError * _Nullable *)outError
{
	if (outError != NULL) {
		*outError = [NFKRemoteTransport errorWithCode:code reason:reason];
	}
	return nil;
}

@end
