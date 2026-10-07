//
//  NFKBalancedBackend.m
//  InferKit
//

#import <InferKit/NFKBalancedBackend.h>
#import <InferKit/NFKRemoteInferKitBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKInferenceServer.h>
#import <InferKit/NFKErrors.h>
#import <InferKit/NFKInferenceKeys.h>
#import "NFKRemoteTransportPrivate.h"
#import <CommonCrypto/CommonDigest.h>

static NSString * const NFKBalancedIdentifier = @"inferkit-balanced";
static const NSTimeInterval NFKBalancedFirstBackoff = 2.0;
static const NSTimeInterval NFKBalancedLongestBackoff = 60.0;
static const NSUInteger NFKBalancedMostConversations = 4096;

/*! A digest naming the messages, or nil when they are not JSON. */
static NSString * _Nullable NFKBalancedMessagesDigest(NSArray *messages)
{
	if (![NSJSONSerialization isValidJSONObject:messages]) {
		return nil;
	}
	NSData *data = [NSJSONSerialization dataWithJSONObject:messages options:NSJSONWritingSortedKeys error:NULL];
	if (data == nil) {
		return nil;
	}
	unsigned char digest[CC_SHA256_DIGEST_LENGTH];
	CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
	NSMutableString *name = [NSMutableString stringWithString:@"messages:"];
	for (NSUInteger index = 0; index < CC_SHA256_DIGEST_LENGTH; index++) {
		[name appendFormat:@"%02x", digest[index]];
	}
	return name;
}

/*! Whether a failure came before the request reached the server: no connection was made at all. */
static BOOL NFKBalancedNeverConnected(NSError *error)
{
	if (![error.domain isEqualToString:NFKInferenceErrorDomain] || error.code != kNFKError_RemoteUnreachable) {
		return NO;
	}
	NSError *underlying = error.userInfo[NSUnderlyingErrorKey];
	NSArray<NSNumber *> *unconnected = @[ @(NSURLErrorCannotConnectToHost), @(NSURLErrorCannotFindHost),
										  @(NSURLErrorDNSLookupFailed), @(NSURLErrorNotConnectedToInternet) ];
	return [underlying.domain isEqualToString:NSURLErrorDomain] && [unconnected containsObject:@(underlying.code)];
}

/*! Whether a failure left the request unstarted, so another server may take it. */
static BOOL NFKBalancedFailsOver(NSError * _Nullable error)
{
	if ([error.domain isEqualToString:NFKInferenceServerErrorDomain]) {
		return error.code == NFKInferenceServerErrorBusy || error.code == NFKInferenceServerErrorModelNotFound;
	}
	if ([error.domain isEqualToString:NFKInferenceErrorDomain] && error.code == kNFKError_InferenceNotReady) {
		return YES;
	}
	return error != nil && NFKBalancedNeverConnected(error);
}

/*! A native client that sends once, so a busy server's refusal reaches the balancer at once rather
	than after the transport's retry delay. */
@interface NFKBalancedClient : NFKRemoteInferKitBackend
@end

@implementation NFKBalancedClient

- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	return [NFKRemoteTransport sendOnce:request session:self.session response:outResponse error:outError];
}

@end

/*! One server and what the balancer knows of it. Every property is read and written on the
	balancer's lock. */
@interface NFKBalancedServer : NSObject
@property (nonatomic, copy) NSURL *baseURL;
@property (nonatomic, copy, nullable) NSString *apiKey;
@property (nonatomic, assign) BOOL discovered;
@property (nonatomic, assign) NSUInteger missedBrowses;
/*! The model's entry in the last status reading, or nil when the server cannot take the model. */
@property (nonatomic, strong, nullable) NFKServerModelStatus *model;
@property (nonatomic, assign) NFKServerThermalState thermalState;
@property (nonatomic, assign) NFKServerMemoryPressure memoryPressure;
@property (nonatomic, copy, nullable) NSDate *statusDate;
@property (nonatomic, strong, nullable) NFKServerLoad *load;
@property (nonatomic, copy, nullable) NSDate *downUntil;
@property (nonatomic, assign) NSTimeInterval backoff;
/*! When each request this balancer sent the server, and that has not finished, was sent. */
@property (nonatomic, strong) NSMutableArray<NSDate *> *dispatches;
@end

@implementation NFKBalancedServer
@end

/*! The server a conversation last ran on. Read and written on the balancer's lock. */
@interface NFKBalancedConversation : NSObject
@property (nonatomic, strong) NFKBalancedServer *server;
@property (nonatomic, copy) NSDate *lastUsed;
@end

@implementation NFKBalancedConversation
@end

@interface NFKBalancedBackend ()
@property (nonatomic, strong) NSMutableArray<NFKBalancedServer *> *servers;
@property (nonatomic, assign) NSUInteger turn;
@property (nonatomic, strong) NSMutableDictionary<NSString *, NFKBalancedConversation *> *conversations;
@property (atomic, readwrite, copy, nullable) NSURL *lastServerBaseURL;
@property (nonatomic, strong) dispatch_queue_t discoveryQueue;
@property (nonatomic, strong, nullable) dispatch_source_t discoveryTimer;
@property (nonatomic, copy, nullable) NSSet<NSString *> *preparedInputKeys;
@property (nonatomic, copy, nullable) NSSet<NSString *> *preparedParameterKeys;
@end

@implementation NFKBalancedBackend

+ (instancetype)backendWithModelName:(nullable NSString *)modelName
{
	NFKBalancedBackend *backend = [[self alloc] init];
	backend.modelName = modelName;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_servers = [NSMutableArray array];
		_statusInterval = 2.0;
		_statusTimeout = 2.0;
		_timeout = 600.0;
		_conversationAffinity = YES;
		_conversationIdleInterval = 600.0;
		_conversationWaitAllowance = 10.0;
		_conversations = [NSMutableDictionary dictionary];
		_discoveryQueue = dispatch_queue_create("com.inferkit.balanced.discovery", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0));
	}
	return self;
}

- (void)dealloc
{
	if (_discoveryTimer != nil) {
		dispatch_source_cancel(_discoveryTimer);
	}
}

#pragma mark Servers

- (NSArray<NSURL *> *)serverBaseURLs
{
	@synchronized (self) {
		return [self.servers valueForKey:@"baseURL"];
	}
}

/*! Runs on the lock. */
- (nullable NFKBalancedServer *)serverWithBaseURL:(NSURL *)baseURL
{
	for (NFKBalancedServer *server in self.servers) {
		if ([server.baseURL isEqual:baseURL]) {
			return server;
		}
	}
	return nil;
}

- (void)addServerWithBaseURL:(NSURL *)baseURL apiKey:(nullable NSString *)apiKey
{
	@synchronized (self) {
		NFKBalancedServer *server = [self serverWithBaseURL:baseURL];
		if (server == nil) {
			server = [[NFKBalancedServer alloc] init];
			server.baseURL = baseURL;
			server.dispatches = [NSMutableArray array];
			[self.servers addObject:server];
		}
		server.apiKey = apiKey;
		server.discovered = NO;
	}
}

- (void)removeServerWithBaseURL:(NSURL *)baseURL
{
	@synchronized (self) {
		NFKBalancedServer *server = [self serverWithBaseURL:baseURL];
		if (server != nil) {
			[self.servers removeObjectIdenticalTo:server];
		}
	}
}

#pragma mark Discovery

- (void)startDiscoveryWithInterval:(NSTimeInterval)interval
{
	[self stopDiscovery];
	NSTimeInterval period = MAX(interval, 1.0);
	NSTimeInterval browse = MIN(MAX(period / 2.0, 0.5), 3.0);
	dispatch_source_t timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, self.discoveryQueue);
	dispatch_source_set_timer(timer, dispatch_time(DISPATCH_TIME_NOW, 0), (uint64_t)(period * NSEC_PER_SEC), (uint64_t)(0.1 * NSEC_PER_SEC));
	__weak NFKBalancedBackend *weakSelf = self;
	dispatch_source_set_event_handler(timer, ^{
		[weakSelf mergeDiscoveredProviders:[NFKRemoteProvider discoverInferKitServersWithTimeout:browse]];
	});
	dispatch_resume(timer);
	@synchronized (self) {
		self.discoveryTimer = timer;
	}
}

- (void)stopDiscovery
{
	dispatch_source_t timer = nil;
	@synchronized (self) {
		timer = self.discoveryTimer;
		self.discoveryTimer = nil;
	}
	if (timer != nil) {
		dispatch_source_cancel(timer);
	}
}

- (void)mergeDiscoveredProviders:(NSArray<NFKRemoteProvider *> *)providers
{
	NSSet<NSURL *> *found = [NSSet setWithArray:[providers valueForKey:@"baseURL"]];
	@synchronized (self) {
		for (NFKBalancedServer *server in [self.servers copy]) {
			if (!server.discovered) {
				continue;
			}
			server.missedBrowses = [found containsObject:server.baseURL] ? 0 : server.missedBrowses + 1;
			if (server.missedBrowses >= 2) {
				[self.servers removeObjectIdenticalTo:server];
			}
		}
		for (NSURL *baseURL in found) {
			if ([self serverWithBaseURL:baseURL] != nil) {
				continue;
			}
			NFKBalancedServer *server = [[NFKBalancedServer alloc] init];
			server.baseURL = baseURL;
			server.discovered = YES;
			server.dispatches = [NSMutableArray array];
			[self.servers addObject:server];
		}
	}
}

#pragma mark Status

- (nullable NFKServerStatus *)fetchStatusFromBaseURL:(NSURL *)baseURL
											 apiKey:(nullable NSString *)apiKey
											  error:(NSError * _Nullable *)outError
{
	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:baseURL];
	client.apiKey = apiKey;
	client.timeout = self.statusTimeout;
	return [client fetchServerStatusWithError:outError];
}

/*! The model's entry a status reading names, or nil when the server cannot take it. */
- (nullable NFKServerModelStatus *)modelInStatus:(NFKServerStatus *)status
{
	NFKServerModelStatus *model = self.modelName.length > 0 ? [status modelNamed:self.modelName]
		: (status.models.count == 1 ? status.models.firstObject : nil);
	if (!model.isReady || [model.backendIdentifier isEqualToString:NFKBalancedIdentifier]) {
		return nil;
	}
	return model;
}

/*! Runs on the lock. */
- (void)markDown:(NFKBalancedServer *)server
{
	server.backoff = server.backoff > 0 ? MIN(server.backoff * 2.0, NFKBalancedLongestBackoff) : NFKBalancedFirstBackoff;
	server.downUntil = [NSDate dateWithTimeIntervalSinceNow:server.backoff];
}

/*! Reads the status of every server whose reading is older than maximumAge and that is not sitting
	out a backoff, all at once, and waits for the readings. */
- (void)refreshStatusesOlderThan:(NSTimeInterval)maximumAge
{
	NSMutableArray<NFKBalancedServer *> *stale = [NSMutableArray array];
	NSMutableArray<NSString *> *keys = [NSMutableArray array];
	@synchronized (self) {
		NSDate *now = [NSDate date];
		for (NFKBalancedServer *server in self.servers) {
			BOOL resting = server.downUntil != nil && [server.downUntil compare:now] == NSOrderedDescending;
			BOOL fresh = server.statusDate != nil && [now timeIntervalSinceDate:server.statusDate] < maximumAge;
			if (!resting && !fresh) {
				[stale addObject:server];
				[keys addObject:server.apiKey ?: self.apiKey ?: @""];
			}
		}
	}
	if (stale.count == 0) {
		return;
	}
	dispatch_group_t group = dispatch_group_create();
	for (NSUInteger index = 0; index < stale.count; index++) {
		NFKBalancedServer *server = stale[index];
		NSURL *baseURL = nil;
		@synchronized (self) {
			baseURL = server.baseURL;
		}
		NSString *key = keys[index].length > 0 ? keys[index] : nil;
		dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			NSError *error = nil;
			NFKServerStatus *status = [self fetchStatusFromBaseURL:baseURL apiKey:key error:&error];
			[self recordStatus:status error:error forServer:server];
		});
	}
	dispatch_group_wait(group, dispatch_time(DISPATCH_TIME_NOW, (int64_t)((self.statusTimeout * 2.0 + 1.0) * NSEC_PER_SEC)));
}

- (void)recordStatus:(nullable NFKServerStatus *)status error:(nullable NSError *)error forServer:(NFKBalancedServer *)server
{
	NFKServerModelStatus *model = status != nil ? [self modelInStatus:status] : nil;
	@synchronized (self) {
		if (status == nil && error != nil && NFKBalancedNeverConnected(error)) {
			[self markDown:server];
			return;
		}
		server.statusDate = [NSDate date];
		server.model = model;
		server.thermalState = status.host.thermalState;
		server.memoryPressure = status.host.memoryPressure;
		if (status != nil) {
			server.downUntil = nil;
			server.backoff = 0;
		}
	}
}

#pragma mark Choosing

/*! A server's load as of its freshest reading: the status route's, or a later reply's headers. */
typedef struct {
	double limit;
	double outstanding;
	double wait;
	BOOL hasWait;
	double averageRun;
	BOOL hasAverage;
} NFKBalancedLoad;

/*! Runs on the lock. */
- (NFKBalancedLoad)loadOf:(NFKBalancedServer *)server
{
	NFKServerModelStatus *model = server.model;
	NFKServerLoad *reply = server.load;
	BOOL replyIsFresher = reply != nil && (server.statusDate == nil || [reply.date compare:server.statusDate] == NSOrderedDescending);
	NSDate *taken = replyIsFresher ? reply.date : server.statusDate;
	NSUInteger sentSince = 0;
	for (NSDate *sent in server.dispatches) {
		sentSince += taken == nil || [sent compare:taken] == NSOrderedDescending ? 1 : 0;
	}
	NFKBalancedLoad load;
	load.limit = MAX((double)(replyIsFresher ? reply.limit : model.limit), 1.0);
	double running = replyIsFresher ? reply.running : model.running;
	double queued = replyIsFresher ? reply.queued : model.queued;
	load.outstanding = running + queued + sentSince;
	NSNumber *wait = replyIsFresher ? reply.estimatedWaitSeconds : model.estimatedWaitSeconds;
	load.hasWait = wait != nil;
	load.wait = wait.doubleValue;
	load.hasAverage = model.averageRunSeconds != nil;
	load.averageRun = model.averageRunSeconds.doubleValue;
	if (load.hasWait && load.hasAverage) {
		load.wait += sentSince * load.averageRun / load.limit;
	} else if (load.hasAverage) {
		load.wait = MAX(load.outstanding - load.limit + 1.0, 0.0) * load.averageRun / load.limit;
		load.hasWait = YES;
	}
	return load;
}

/*! Runs on the lock. */
- (BOOL)isStrained:(NFKBalancedServer *)server
{
	NFKServerThermalState thermal = server.load != nil && server.load.thermalState != NFKServerThermalStateUnknown
		&& (server.statusDate == nil || [server.load.date compare:server.statusDate] == NSOrderedDescending)
		? server.load.thermalState : server.thermalState;
	return thermal >= NFKServerThermalStateSerious || server.memoryPressure == NFKServerMemoryPressureCritical;
}

/*! Runs on the lock. */
- (NFKBalancedServer *)pickAmong:(NSArray<NFKBalancedServer *> *)candidates
{
	NSUInteger start = self.turn % candidates.count;
	self.turn += 1;
	NSMutableArray<NFKBalancedServer *> *rotated = [NSMutableArray arrayWithCapacity:candidates.count];
	for (NSUInteger offset = 0; offset < candidates.count; offset++) {
		[rotated addObject:candidates[(start + offset) % candidates.count]];
	}
	if (self.policy == NFKBalancingPolicyRoundRobin) {
		return rotated.firstObject;
	}
	BOOL byWait = self.policy == NFKBalancingPolicyShortestExpectedWait;
	for (NFKBalancedServer *server in rotated) {
		NFKBalancedLoad load = [self loadOf:server];
		byWait = byWait && load.hasWait;
	}
	NFKBalancedServer *best = nil;
	double bestScore = INFINITY;
	for (NFKBalancedServer *server in rotated) {
		NFKBalancedLoad load = [self loadOf:server];
		double score = byWait ? load.wait : load.outstanding / load.limit;
		if (score < bestScore) {
			best = server;
			bestScore = score;
		}
	}
	return best ?: rotated.firstObject;
}

#pragma mark Conversations

/*! The name of the request's conversation: its NFKParameterConversationKey, or a digest of its messages
	up to and including the first user message. Nil when the request has neither, or when affinity
	is off. */
- (nullable NSString *)conversationOfRequest:(NFKInferenceRequest *)request
{
	if (!self.conversationAffinity) {
		return nil;
	}
	id named = request.parameters[NFKParameterConversationKey];
	if ([named isKindOfClass:NSString.class] && [(NSString *)named length] > 0) {
		return [@"key:" stringByAppendingString:named];
	}
	id messages = request.inputs[NFKInputMessages];
	if (![messages isKindOfClass:NSArray.class]) {
		return nil;
	}
	NSMutableArray *opening = [NSMutableArray array];
	for (id message in messages) {
		[opening addObject:message];
		if ([message isKindOfClass:NSDictionary.class] && [message[@"role"] isEqual:@"user"]) {
			return NFKBalancedMessagesDigest(opening);
		}
	}
	return nil;
}

/*! Runs on the lock. The server the conversation last ran on, forgetting it once idle too long. */
- (nullable NFKBalancedServer *)serverOfConversation:(nullable NSString *)conversation now:(NSDate *)now
{
	if (conversation == nil) {
		return nil;
	}
	NFKBalancedConversation *entry = self.conversations[conversation];
	if (entry != nil && [now timeIntervalSinceDate:entry.lastUsed] > self.conversationIdleInterval) {
		[self.conversations removeObjectForKey:conversation];
		return nil;
	}
	return entry.server;
}

/*! Runs on the lock. Whether a conversation stays on its server: it does unless a candidate's
	expected wait is more than conversationWaitAllowance shorter than its server's. */
- (BOOL)keepsConversationOn:(NFKBalancedServer *)server among:(NSArray<NFKBalancedServer *> *)candidates
{
	NFKBalancedLoad own = [self loadOf:server];
	if (!own.hasWait) {
		return YES;
	}
	for (NFKBalancedServer *candidate in candidates) {
		NFKBalancedLoad load = [self loadOf:candidate];
		if (load.hasWait && own.wait - load.wait > self.conversationWaitAllowance) {
			return NO;
		}
	}
	return YES;
}

/*! The request as sent to a server. A conversation the balancer named from the request's messages
	travels under NFKParameterConversationKey, so the server's backend keeps that conversation's
	prompt as it would for a caller's own key. */
- (NFKInferenceRequest *)request:(NFKInferenceRequest *)request forwardingConversation:(nullable NSString *)conversation
{
	if (conversation == nil || request.parameters[NFKParameterConversationKey] != nil) {
		return request;
	}
	NSMutableDictionary<NSString *, id> *parameters = [request.parameters mutableCopy] ?: [NSMutableDictionary dictionary];
	parameters[NFKParameterConversationKey] = conversation;
	return [NFKInferenceRequest requestWithInputs:request.inputs parameters:parameters outputModality:request.outputModality];
}

/*! Ties the conversation to the server that answered it. */
- (void)recordConversation:(nullable NSString *)conversation onServer:(NFKBalancedServer *)server
{
	if (conversation == nil) {
		return;
	}
	@synchronized (self) {
		NFKBalancedConversation *entry = self.conversations[conversation] ?: [[NFKBalancedConversation alloc] init];
		entry.server = server;
		entry.lastUsed = [NSDate date];
		self.conversations[conversation] = entry;
		if (self.conversations.count > NFKBalancedMostConversations) {
			[self forgetLeastRecentConversations];
		}
	}
}

/*! Runs on the lock. Keeps the most recently used conversations, up to the bound. */
- (void)forgetLeastRecentConversations
{
	NSArray<NSString *> *ordered = [self.conversations keysSortedByValueUsingComparator:^NSComparisonResult(NFKBalancedConversation *a, NFKBalancedConversation *b) {
		return [a.lastUsed compare:b.lastUsed];
	}];
	NSUInteger excess = ordered.count - NFKBalancedMostConversations;
	[self.conversations removeObjectsForKeys:[ordered subarrayWithRange:NSMakeRange(0, excess)]];
}

#pragma mark Choosing a server

/*! The server for the next attempt, or nil when no untried server can take the model. A strained
	server is chosen only when no other candidate remains. A conversation stays on its server while
	keepsConversationOn:among: holds. */
- (nullable NFKBalancedServer *)chooseServerExcluding:(NSSet<NFKBalancedServer *> *)tried
										 conversation:(nullable NSString *)conversation
{
	[self refreshStatusesOlderThan:self.statusInterval];
	@synchronized (self) {
		NSDate *now = [NSDate date];
		NSMutableArray<NFKBalancedServer *> *healthy = [NSMutableArray array];
		NSMutableArray<NFKBalancedServer *> *strained = [NSMutableArray array];
		for (NFKBalancedServer *server in self.servers) {
			BOOL resting = server.downUntil != nil && [server.downUntil compare:now] == NSOrderedDescending;
			if ([tried containsObject:server] || resting || server.model == nil) {
				continue;
			}
			[[self isStrained:server] ? strained : healthy addObject:server];
		}
		NSArray<NFKBalancedServer *> *candidates = healthy.count > 0 ? healthy : strained;
		if (candidates.count == 0) {
			return nil;
		}
		NFKBalancedServer *own = [self serverOfConversation:conversation now:now];
		if (own != nil && [candidates containsObject:own] && [self keepsConversationOn:own among:candidates]) {
			return own;
		}
		return [self pickAmong:candidates];
	}
}

#pragma mark Dispatch

- (NFKRemoteInferKitBackend *)clientForServer:(NFKBalancedServer *)server sentAt:(NSDate *)sent
{
	NFKBalancedClient *client = nil;
	@synchronized (self) {
		client = [NFKBalancedClient backendWithBaseURL:server.baseURL];
		client.apiKey = server.apiKey ?: self.apiKey;
		[server.dispatches addObject:sent];
	}
	client.modelName = self.modelName;
	client.timeout = self.timeout;
	return client;
}

/*! Records how a request sent to a server ended. */
- (void)server:(NFKBalancedServer *)server finished:(NSDate *)sent client:(NFKRemoteInferKitBackend *)client error:(nullable NSError *)error
{
	@synchronized (self) {
		[server.dispatches removeObjectIdenticalTo:sent];
		if (client.lastReportedLoad != nil) {
			server.load = client.lastReportedLoad;
		}
		if (error == nil) {
			return;
		}
		if (NFKBalancedNeverConnected(error)) {
			[self markDown:server];
		} else if (NFKBalancedFailsOver(error) && !(error.code == NFKInferenceServerErrorBusy
													 && [error.domain isEqualToString:NFKInferenceServerErrorDomain])) {
			server.model = nil;
		}
	}
}

- (NSURL *)baseURLOfServer:(NFKBalancedServer *)server
{
	@synchronized (self) {
		return server.baseURL;
	}
}

- (NSError *)noServerError:(nullable NSError *)lastError
{
	if (lastError != nil) {
		return lastError;
	}
	NSString *model = self.modelName.length > 0 ? [NSString stringWithFormat:@"the model \"%@\"", self.modelName] : @"the model";
	return [NFKRemoteTransport errorWithCode:kNFKError_InferenceNotReady
									  reason:[NSString stringWithFormat:@"no reachable server can take %@ now", model]];
}

#pragma mark NFKInferenceBackend

- (BOOL)isReady
{
	@synchronized (self) {
		return self.servers.count > 0;
	}
}

- (NSString *)backendIdentifier
{
	return NFKBalancedIdentifier;
}

- (BOOL)respondsToSelector:(SEL)selector
{
	if (selector == @selector(supportedInputKeys)) {
		return self.preparedInputKeys != nil;
	}
	if (selector == @selector(supportedParameterKeys)) {
		return self.preparedParameterKeys != nil;
	}
	return [super respondsToSelector:selector];
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return self.preparedInputKeys ?: [NSSet set];
}

- (NSSet<NSString *> *)supportedParameterKeys
{
	return self.preparedParameterKeys ?: [NSSet set];
}

- (NSDictionary<NSString *, id> *)modelInfo
{
	@synchronized (self) {
		for (NFKBalancedServer *server in self.servers) {
			if (server.model != nil) {
				return server.model.modelInfo;
			}
		}
	}
	return @{};
}

- (BOOL)prepareWithError:(NSError * _Nullable *)outError
{
	[self refreshStatusesOlderThan:0];
	NFKBalancedServer *server = [self chooseServerExcluding:[NSSet set] conversation:nil];
	if (server == nil) {
		if (outError != NULL) {
			*outError = [self noServerError:nil];
		}
		return NO;
	}
	NSDate *sent = [NSDate date];
	NFKRemoteInferKitBackend *client = [self clientForServer:server sentAt:sent];
	NSError *error = nil;
	BOOL prepared = [client prepareWithError:&error];
	[self server:server finished:sent client:client error:prepared ? nil : error];
	if (prepared) {
		self.preparedInputKeys = client.supportedInputKeys;
		self.preparedParameterKeys = client.supportedParameterKeys;
	} else if (outError != NULL) {
		*outError = error;
	}
	return prepared;
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError * _Nullable *)outError
{
	NSString *conversation = [self conversationOfRequest:request];
	NFKInferenceRequest *forwarded = [self request:request forwardingConversation:conversation];
	NSMutableSet<NFKBalancedServer *> *tried = [NSMutableSet set];
	NSError *lastError = nil;
	for (NFKBalancedServer *server = [self chooseServerExcluding:tried conversation:conversation]; server != nil;
		 server = [self chooseServerExcluding:tried conversation:conversation]) {
		[tried addObject:server];
		NSDate *sent = [NSDate date];
		NFKRemoteInferKitBackend *client = [self clientForServer:server sentAt:sent];
		NSError *error = nil;
		NFKInferenceResult *result = [client runInferenceForRequest:forwarded error:&error];
		[self server:server finished:sent client:client error:result != nil ? nil : error];
		if (result != nil) {
			[self recordConversation:conversation onServer:server];
			self.lastServerBaseURL = [self baseURLOfServer:server];
			return result;
		}
		lastError = error;
		if (!NFKBalancedFailsOver(error)) {
			break;
		}
	}
	if (outError != NULL) {
		*outError = [self noServerError:lastError];
	}
	return nil;
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	NSString *conversation = [self conversationOfRequest:request];
	[self submitRequest:[self request:request forwardingConversation:conversation] conversation:conversation
				  toJob:job tried:[NSMutableSet set] lastError:nil];
	return job;
}

/*! Sends a streamed request to the next server and fails over while the run has not started. */
- (void)submitRequest:(NFKInferenceRequest *)request
		 conversation:(nullable NSString *)conversation
				toJob:(NFKInferenceJob *)job
				tried:(NSMutableSet<NFKBalancedServer *> *)tried
			lastError:(nullable NSError *)lastError
{
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		if (job.status == NFKInferenceJobStatusCancelled) {
			return;
		}
		NFKBalancedServer *server = [self chooseServerExcluding:tried conversation:conversation];
		if (server == nil) {
			[job finishWithError:[self noServerError:lastError]];
			return;
		}
		[tried addObject:server];
		NSDate *sent = [NSDate date];
		NFKRemoteInferKitBackend *client = [self clientForServer:server sentAt:sent];
		NFKInferenceJob *inner = [client submitInferenceJobForRequest:request];
		__block BOOL started = NO;
		inner.progressHandler = ^(NFKInferenceJob *progressed) {
			started = started || progressed.partialResult != nil || progressed.progress > 0;
			[job reportProgress:progressed.progress partialResult:progressed.partialResult];
		};
		inner.completionHandler = ^(NFKInferenceJob *finished) {
			NSError *error = finished.status == NFKInferenceJobStatusSucceeded ? nil : finished.error;
			[self server:server finished:sent client:client error:error];
			if (finished.status == NFKInferenceJobStatusSucceeded) {
				[self recordConversation:conversation onServer:server];
				self.lastServerBaseURL = [self baseURLOfServer:server];
				[job finishWithResult:finished.result];
			} else if (finished.status == NFKInferenceJobStatusCancelled || job.status == NFKInferenceJobStatusCancelled) {
				return;
			} else if (!started && NFKBalancedFailsOver(error)) {
				[self submitRequest:request conversation:conversation toJob:job tried:tried lastError:error];
			} else {
				[job finishWithError:error ?: [self noServerError:nil]];
			}
		};
		job.cancellationHandler = ^{
			[inner cancel];
		};
		// A cancel that landed before the handler was installed ran no handler.
		if (job.status == NFKInferenceJobStatusCancelled) {
			[inner cancel];
		}
	});
}

@end
