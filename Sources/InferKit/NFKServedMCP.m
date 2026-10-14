//
//  NFKServedMCP.m
//  InferKit
//

#import "NFKServedMCP.h"
#import "NFKServerObservationPrivate.h"
#import <InferKit/NFKInferKit.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKAudioAsset.h>
#import <InferKit/NFKVideoAsset.h>

static NSInteger const NFKJSONRPCParseError		= -32700;
static NSInteger const NFKJSONRPCInvalidRequest	= -32600;
static NSInteger const NFKJSONRPCMethodNotFound	= -32601;
static NSInteger const NFKJSONRPCInvalidParams	= -32602;

NSArray<NSString *> *NFKMCPProtocolVersions(void)
{
	return @[ @"2025-11-25", @"2025-06-18", @"2025-03-26", @"2024-11-05" ];
}

static NSString *NFKMCPMediaType(NSURL *fileURL, NSString *fallback)
{
	NSDictionary<NSString *, NSString *> *types = @{ @"wav": @"audio/wav", @"mp3": @"audio/mpeg", @"m4a": @"audio/mp4",
													 @"aac": @"audio/aac", @"flac": @"audio/flac", @"aif": @"audio/aiff",
													 @"aiff": @"audio/aiff", @"caf": @"audio/x-caf", @"ogg": @"audio/ogg",
													 @"mp4": @"video/mp4", @"mov": @"video/quicktime", @"m4v": @"video/mp4",
													 @"webm": @"video/webm" };
	return types[fileURL.pathExtension.lowercaseString] ?: fallback;
}

NSDictionary<NSString *, id> *NFKMCPCallResult(NFKToolResult *result)
{
	NSMutableArray *content = [NSMutableArray arrayWithObject:@{ @"type": @"text", @"text": result.text }];
	for (NSData *png in result.imagePNGs) {
		[content addObject:@{ @"type": @"image", @"data": [png base64EncodedStringWithOptions:0], @"mimeType": @"image/png" }];
	}
	id audio = result.outputs[NFKOutputAudio];
	NSURL *audioURL = [audio isKindOfClass:NFKAudioAsset.class] ? [(NFKAudioAsset *)audio fileURL] : nil;
	NSData *audioData = audioURL != nil ? [NSData dataWithContentsOfURL:audioURL] : nil;
	if (audioData != nil) {
		[content addObject:@{ @"type": @"audio", @"data": [audioData base64EncodedStringWithOptions:0],
							  @"mimeType": NFKMCPMediaType(audioURL, @"audio/wav") }];
	}
	id video = result.outputs[NFKOutputVideo];
	NSURL *videoURL = [video isKindOfClass:NFKVideoAsset.class] ? [(NFKVideoAsset *)video fileURL] : nil;
	NSData *videoData = videoURL != nil ? [NSData dataWithContentsOfURL:videoURL] : nil;
	if (videoData != nil) {
		[content addObject:@{ @"type": @"resource",
							  @"resource": @{ @"uri": [@"inferkit:///" stringByAppendingString:videoURL.lastPathComponent],
											  @"mimeType": NFKMCPMediaType(videoURL, @"video/mp4"),
											  @"blob": [videoData base64EncodedStringWithOptions:0] } }];
	}
	NSMutableDictionary *reply = [NSMutableDictionary dictionaryWithDictionary:@{ @"content": content, @"isError": @(result.isError) }];
	id structured = result.outputs[NFKOutputStructured];
	if ([structured isKindOfClass:NSDictionary.class] && [NSJSONSerialization isValidJSONObject:structured]) {
		reply[@"structuredContent"] = structured;
	}
	return reply;
}

#pragma mark - Sessions

static NSString * const NFKMCPPingPrefix = @"inferkit-ping-";

static NSTimeInterval NFKMCPNow(void)
{
	return NSProcessInfo.processInfo.systemUptime;
}

/*! One session's state. Every property is read and written on the store's lock. */
@interface NFKServedMCPSessionRecord : NSObject
@property (nonatomic, copy) NSString *identifier;
@property (nonatomic, copy, nullable) NSString *clientName;
@property (nonatomic, copy, nullable) NSString *clientVersion;
@property (nonatomic, copy) NSString *protocolVersion;
@property (nonatomic, copy) NSString *clientAddress;
@property (nonatomic, copy, nullable) NSString *keyName;
@property (nonatomic, copy) NSDate *startDate;
@property (nonatomic, assign) NSTimeInterval startedAt;
@property (nonatomic, assign) NSTimeInterval lastActivity;
@property (nonatomic, strong, nullable) NFKHTTPResponse *stream;
@property (nonatomic, assign) NSTimeInterval lastPingAt;
@property (nonatomic, assign) NSUInteger outstandingPings;
@property (nonatomic, assign) NSUInteger pingsSent;
@property (nonatomic, assign) NSUInteger inFlight;
@end

@implementation NFKServedMCPSessionRecord

- (NFKServerMCPSession *)snapshotAt:(NSTimeInterval)now endReason:(NFKServerMCPSessionEndReason)reason
{
	NFKServerMCPSession *session = [[NFKServerMCPSession alloc] initPrivately];
	session.identifier = self.identifier;
	session.clientName = self.clientName;
	session.clientVersion = self.clientVersion;
	session.protocolVersion = self.protocolVersion;
	session.clientAddress = self.clientAddress;
	session.keyName = self.keyName;
	session.startDate = self.startDate;
	session.lastActivityDate = [NSDate dateWithTimeIntervalSinceNow:-MAX(now - self.lastActivity, 0.0)];
	session.streamOpen = self.stream != nil;
	session.endReason = reason;
	session.endDate = reason != NFKServerMCPSessionEndReasonNone ? [NSDate date] : nil;
	return session;
}

@end

@interface NFKServedMCPSessionStore ()
@property (nonatomic, strong) NSMutableDictionary<NSString *, NFKServedMCPSessionRecord *> *records;
@property (nonatomic, strong) dispatch_queue_t queue;
/*! Delivers the event handler's calls in order. */
@property (nonatomic, strong) dispatch_queue_t eventQueue;
@property (nonatomic, strong, nullable) dispatch_source_t timer;
@end

@implementation NFKServedMCPSessionStore

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_records = [NSMutableDictionary dictionary];
		_idleTimeout = 1800;
		_pingInterval = 30;
		_missedPingLimit = 3;
		_queue = dispatch_queue_create("NFKInferenceServer.mcp", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0));
		_eventQueue = dispatch_queue_create("NFKInferenceServer.mcp.events", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0));
	}
	return self;
}

- (void)dealloc
{
	if (_timer != nil) {
		dispatch_source_cancel(_timer);
	}
}

- (void)report:(NFKServerMCPSession *)session ended:(BOOL)ended
{
	void (^handler)(NFKServerMCPSession *, BOOL) = self.eventHandler;
	if (handler == nil) {
		return;
	}
	dispatch_async(self.eventQueue, ^{
		handler(session, ended);
	});
}

#pragma mark Lifecycle

- (NFKServedMCPSessionRecord *)beginSessionForRequest:(NFKHTTPRequest *)request
										   clientInfo:(nullable NSDictionary *)clientInfo
									  protocolVersion:(NSString *)protocolVersion
{
	NSTimeInterval now = NFKMCPNow();
	NFKServedMCPSessionRecord *record = [[NFKServedMCPSessionRecord alloc] init];
	record.identifier = NSUUID.UUID.UUIDString;
	record.clientName = [clientInfo[@"name"] isKindOfClass:NSString.class] ? clientInfo[@"name"] : nil;
	record.clientVersion = [clientInfo[@"version"] isKindOfClass:NSString.class] ? clientInfo[@"version"] : nil;
	record.protocolVersion = protocolVersion;
	record.clientAddress = request.clientAddress ?: @"";
	record.keyName = request.keyName;
	record.startDate = [NSDate date];
	record.startedAt = now;
	record.lastActivity = now;
	NFKServerMCPSession *snapshot = nil;
	@synchronized (self) {
		self.records[record.identifier] = record;
		snapshot = [record snapshotAt:now endReason:NFKServerMCPSessionEndReasonNone];
		[self scheduleTimer];
	}
	[self report:snapshot ended:NO];
	return record;
}

/*! The active session a request names, when it started with the same key name, or nil. */
- (nullable NFKServedMCPSessionRecord *)activeSessionWithIdentifier:(NSString *)identifier keyName:(nullable NSString *)keyName
{
	@synchronized (self) {
		NFKServedMCPSessionRecord *record = self.records[identifier];
		BOOL sameKey = record.keyName == keyName || [record.keyName isEqualToString:keyName];
		return sameKey ? record : nil;
	}
}

- (BOOL)endSessionWithIdentifier:(NSString *)identifier reason:(NFKServerMCPSessionEndReason)reason
{
	NFKServedMCPSessionRecord *record = nil;
	@synchronized (self) {
		record = self.records[identifier];
	}
	return record != nil && [self endSession:record reason:reason];
}

/*! Removes the session and closes its stream; NO when it had already ended. */
- (BOOL)endSession:(NFKServedMCPSessionRecord *)record reason:(NFKServerMCPSessionEndReason)reason
{
	NFKHTTPResponse *stream = nil;
	NFKServerMCPSession *snapshot = nil;
	@synchronized (self) {
		if (self.records[record.identifier] != record) {
			return NO;
		}
		[self.records removeObjectForKey:record.identifier];
		snapshot = [record snapshotAt:NFKMCPNow() endReason:reason];
		stream = record.stream;
		record.stream = nil;
		if (self.records.count == 0) {
			[self cancelTimer];
		}
	}
	[stream endStream];
	[self report:snapshot ended:YES];
	return YES;
}

- (void)endSessionsWithKeyName:(NSString *)keyName reason:(NFKServerMCPSessionEndReason)reason
{
	NSArray<NFKServedMCPSessionRecord *> *matching = nil;
	@synchronized (self) {
		matching = [self.records.allValues filteredArrayUsingPredicate:[NSPredicate predicateWithBlock:^BOOL(NFKServedMCPSessionRecord *record, NSDictionary *bindings) {
			return [record.keyName isEqualToString:keyName];
		}]];
	}
	for (NFKServedMCPSessionRecord *record in matching) {
		[self endSession:record reason:reason];
	}
}

- (void)endAllSessionsWithReason:(NFKServerMCPSessionEndReason)reason
{
	NSArray<NFKServedMCPSessionRecord *> *all = nil;
	@synchronized (self) {
		all = [self.records.allValues copy];
	}
	for (NFKServedMCPSessionRecord *record in all) {
		[self endSession:record reason:reason];
	}
}

- (NSArray<NFKServerMCPSession *> *)sessions
{
	NSTimeInterval now = NFKMCPNow();
	@synchronized (self) {
		NSArray<NFKServedMCPSessionRecord *> *oldestFirst = [self.records.allValues sortedArrayUsingComparator:^NSComparisonResult(NFKServedMCPSessionRecord *left, NFKServedMCPSessionRecord *right) {
			return [@(left.startedAt) compare:@(right.startedAt)];
		}];
		NSMutableArray<NFKServerMCPSession *> *snapshots = [NSMutableArray arrayWithCapacity:oldestFirst.count];
		for (NFKServedMCPSessionRecord *record in oldestFirst) {
			[snapshots addObject:[record snapshotAt:now endReason:NFKServerMCPSessionEndReasonNone]];
		}
		return snapshots;
	}
}

#pragma mark Activity

- (void)beginRequestInSession:(nullable NFKServedMCPSessionRecord *)record
{
	@synchronized (self) {
		record.inFlight += 1;
		record.lastActivity = NFKMCPNow();
	}
}

- (void)endRequestInSession:(nullable NFKServedMCPSessionRecord *)record
{
	@synchronized (self) {
		record.inFlight -= record.inFlight > 0 ? 1 : 0;
		record.lastActivity = NFKMCPNow();
	}
}

/*! Records a client's answer to one of the server's pings. */
- (void)receivedResponseWithIdentifier:(id)identifier inSession:(nullable NFKServedMCPSessionRecord *)record
{
	if (![identifier isKindOfClass:NSString.class] || ![identifier hasPrefix:NFKMCPPingPrefix]) {
		return;
	}
	@synchronized (self) {
		record.outstandingPings = 0;
		record.lastActivity = NFKMCPNow();
	}
}

#pragma mark Streams

/*! Makes response the session's event stream, closing the one it replaces; NO when the session has
	ended. */
- (BOOL)openStream:(NFKHTTPResponse *)response forSession:(NFKServedMCPSessionRecord *)record
{
	NFKHTTPResponse *replaced = nil;
	@synchronized (self) {
		if (self.records[record.identifier] != record) {
			return NO;
		}
		replaced = record.stream;
		record.stream = response;
		record.outstandingPings = 0;
		record.lastPingAt = NFKMCPNow();
		record.lastActivity = record.lastPingAt;
	}
	__weak NFKServedMCPSessionStore *weakSelf = self;
	__weak NFKServedMCPSessionRecord *weakRecord = record;
	__weak NFKHTTPResponse *weakResponse = response;
	response.disconnectHandler = ^{
		[weakSelf streamClosed:weakResponse inSession:weakRecord];
	};
	[response beginStreamWithStatus:200 headers:@{ @"Content-Type": @"text/event-stream", @"Mcp-Session-Id": record.identifier }];
	[replaced endStream];
	return YES;
}

- (void)streamClosed:(nullable NFKHTTPResponse *)response inSession:(nullable NFKServedMCPSessionRecord *)record
{
	@synchronized (self) {
		if (response != nil && record.stream == response) {
			record.stream = nil;
			record.outstandingPings = 0;
			record.lastActivity = NFKMCPNow();
		}
	}
}

#pragma mark Expiry

- (void)settingsChanged
{
	@synchronized (self) {
		if (self.timer != nil) {
			[self cancelTimer];
			[self scheduleTimer];
		}
	}
}

/*! A quarter of the shortest enabled interval, within 10 ms and 1 s, or 0 when neither is enabled. */
- (NSTimeInterval)tickInterval
{
	NSTimeInterval shortest = INFINITY;
	if (self.pingInterval > 0) {
		shortest = MIN(shortest, self.pingInterval);
	}
	if (self.idleTimeout > 0) {
		shortest = MIN(shortest, self.idleTimeout);
	}
	return isfinite(shortest) ? MIN(MAX(shortest / 4.0, 0.01), 1.0) : 0;
}

/*! Runs on the lock. */
- (void)scheduleTimer
{
	NSTimeInterval tick = [self tickInterval];
	if (self.timer != nil || tick <= 0) {
		return;
	}
	dispatch_source_t timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, self.queue);
	uint64_t nanoseconds = (uint64_t)(tick * NSEC_PER_SEC);
	dispatch_source_set_timer(timer, dispatch_time(DISPATCH_TIME_NOW, (int64_t)nanoseconds), nanoseconds, nanoseconds / 10);
	__weak NFKServedMCPSessionStore *weakSelf = self;
	dispatch_source_set_event_handler(timer, ^{
		[weakSelf sweep];
	});
	dispatch_resume(timer);
	self.timer = timer;
}

/*! Runs on the lock. */
- (void)cancelTimer
{
	if (self.timer != nil) {
		dispatch_source_cancel(self.timer);
		self.timer = nil;
	}
}

/*! Pings each open stream that is due one and ends the sessions that missed too many pings or sat idle. */
- (void)sweep
{
	NSTimeInterval now = NFKMCPNow();
	NSTimeInterval pingInterval = self.pingInterval;
	NSTimeInterval idleTimeout = self.idleTimeout;
	NSUInteger missedLimit = MAX(self.missedPingLimit, (NSUInteger)1);
	NSMutableArray<NFKServedMCPSessionRecord *> *missed = [NSMutableArray array];
	NSMutableArray<NFKServedMCPSessionRecord *> *idle = [NSMutableArray array];
	NSMutableArray<NSArray *> *pings = [NSMutableArray array];
	@synchronized (self) {
		for (NFKServedMCPSessionRecord *record in self.records.allValues) {
			if (record.stream != nil) {
				if (pingInterval <= 0 || now - record.lastPingAt < pingInterval) {
					continue;
				}
				if (record.outstandingPings >= missedLimit) {
					[missed addObject:record];
					continue;
				}
				record.outstandingPings += 1;
				record.pingsSent += 1;
				record.lastPingAt = now;
				NSString *identifier = [NSString stringWithFormat:@"%@%lu", NFKMCPPingPrefix, (unsigned long)record.pingsSent];
				[pings addObject:@[ record.stream, identifier ]];
			} else if (idleTimeout > 0 && record.inFlight == 0 && now - record.lastActivity >= idleTimeout) {
				[idle addObject:record];
			}
		}
	}
	for (NSArray *ping in pings) {
		NSDictionary *message = @{ @"jsonrpc": @"2.0", @"id": ping[1], @"method": @"ping" };
		NSData *data = [NSJSONSerialization dataWithJSONObject:message options:0 error:NULL];
		[(NFKHTTPResponse *)ping[0] sendEventData:[[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding]];
	}
	for (NFKServedMCPSessionRecord *record in missed) {
		[self endSession:record reason:NFKServerMCPSessionEndReasonPingTimeout];
	}
	for (NFKServedMCPSessionRecord *record in idle) {
		[self endSession:record reason:NFKServerMCPSessionEndReasonIdle];
	}
}

@end

#pragma mark - Endpoint

@implementation NFKServedMCP

+ (void)serveRequest:(NFKHTTPRequest *)request
			response:(NFKHTTPResponse *)response
			   tools:(NSArray<NFKInferenceTool *> *)tools
		instructions:(nullable NSString *)instructions
			sessions:(NFKServedMCPSessionStore *)sessions
{
	if (![self originIsAllowedFor:request]) {
		[response sendStatus:403 JSONObject:[self errorWithIdentifier:nil code:NFKJSONRPCInvalidRequest
															  message:@"the Origin header names another host"] headers:nil];
		return;
	}
	NSString *version = [request valueForHeader:@"mcp-protocol-version"];
	if (version != nil && ![NFKMCPProtocolVersions() containsObject:version]) {
		[response sendStatus:400 JSONObject:[self errorWithIdentifier:nil code:NFKJSONRPCInvalidRequest
															  message:[@"unsupported MCP-Protocol-Version " stringByAppendingString:version]] headers:nil];
		return;
	}
	NSString *sessionIdentifier = [request valueForHeader:@"mcp-session-id"];
	if ([request.method isEqualToString:@"GET"] || [request.method isEqualToString:@"DELETE"]) {
		[self serveSessionRequest:request identifier:sessionIdentifier response:response sessions:sessions];
		return;
	}
	if (![request.method isEqualToString:@"POST"]) {
		[response sendStatus:405 headers:@{ @"Allow": @"GET, POST, DELETE" } body:nil];
		return;
	}
	id body = request.body.length > 0 ? [NSJSONSerialization JSONObjectWithData:request.body options:0 error:NULL] : nil;
	if (body == nil) {
		[response sendStatus:400 JSONObject:[self errorWithIdentifier:nil code:NFKJSONRPCParseError message:@"the body is not JSON"] headers:nil];
		return;
	}
	BOOL batch = [body isKindOfClass:NSArray.class];
	NSArray *messages = batch ? body : @[ body ];
	BOOL initializes = NO;
	for (id message in messages) {
		initializes = initializes || ([message isKindOfClass:NSDictionary.class] && [message[@"method"] isEqual:@"initialize"] && message[@"id"] != nil);
	}
	NFKServedMCPSessionRecord *session = nil;
	if (!initializes && sessionIdentifier != nil) {
		session = [sessions activeSessionWithIdentifier:sessionIdentifier keyName:request.keyName];
		if (session == nil) {
			[self sendUnknownSession:response];
			return;
		}
	}
	[sessions beginRequestInSession:session];
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NSMutableArray *replies = [NSMutableArray array];
		NFKServedMCPSessionRecord *issued = nil;
		for (id message in messages) {
			if ([self isResponse:message]) {
				[sessions receivedResponseWithIdentifier:message[@"id"] inSession:session];
				continue;
			}
			NSDictionary *reply = [self replyTo:message tools:tools instructions:instructions];
			if (reply == nil) {
				continue;
			}
			[replies addObject:reply];
			NSDictionary *result = reply[@"result"];
			if (issued == nil && [message[@"method"] isEqual:@"initialize"] && [result isKindOfClass:NSDictionary.class]) {
				NSDictionary *params = [message[@"params"] isKindOfClass:NSDictionary.class] ? message[@"params"] : @{};
				NSDictionary *clientInfo = [params[@"clientInfo"] isKindOfClass:NSDictionary.class] ? params[@"clientInfo"] : nil;
				issued = [sessions beginSessionForRequest:request clientInfo:clientInfo protocolVersion:result[@"protocolVersion"]];
			}
		}
		[sessions endRequestInSession:session];
		NSDictionary *headers = issued != nil ? @{ @"Mcp-Session-Id": issued.identifier } : nil;
		if (replies.count == 0) {
			[response sendStatus:202 headers:headers body:nil];
			return;
		}
		[response sendStatus:200 JSONObject:batch ? replies : replies.firstObject headers:headers];
	});
}

/*! GET opens the session's event stream and DELETE ends the session. */
+ (void)serveSessionRequest:(NFKHTTPRequest *)request
				 identifier:(nullable NSString *)identifier
				   response:(NFKHTTPResponse *)response
				   sessions:(NFKServedMCPSessionStore *)sessions
{
	BOOL opensStream = [request.method isEqualToString:@"GET"];
	if (identifier == nil) {
		if (opensStream) {
			[response sendStatus:405 headers:@{ @"Allow": @"GET, POST, DELETE" } body:nil];
		} else {
			[response sendStatus:400 JSONObject:[self errorWithIdentifier:nil code:NFKJSONRPCInvalidRequest
																  message:@"DELETE names its session in Mcp-Session-Id"] headers:nil];
		}
		return;
	}
	NFKServedMCPSessionRecord *session = [sessions activeSessionWithIdentifier:identifier keyName:request.keyName];
	if (session == nil) {
		[self sendUnknownSession:response];
		return;
	}
	if (opensStream) {
		if (![sessions openStream:response forSession:session]) {
			[self sendUnknownSession:response];
		}
		return;
	}
	[sessions endSession:session reason:NFKServerMCPSessionEndReasonDeleted];
	[response sendStatus:204 headers:nil body:nil];
}

+ (void)sendUnknownSession:(NFKHTTPResponse *)response
{
	[response sendStatus:404 JSONObject:[self errorWithIdentifier:nil code:NFKJSONRPCInvalidRequest
														  message:@"the session is unknown or has ended; initialize a new one"] headers:nil];
}

/*! Whether message is a JSON-RPC response, which carries an id and a result or an error but no method. */
+ (BOOL)isResponse:(id)message
{
	return [message isKindOfClass:NSDictionary.class] && message[@"method"] == nil && message[@"id"] != nil
		&& (message[@"result"] != nil || message[@"error"] != nil);
}

// A browser sends Origin; a page on another host that resolves to this machine names its own.
+ (BOOL)originIsAllowedFor:(NFKHTTPRequest *)request
{
	NSString *origin = [request valueForHeader:@"origin"];
	if (origin.length == 0 || [origin isEqualToString:@"null"]) {
		return origin.length == 0;
	}
	NSString *host = [NSURL URLWithString:origin].host.lowercaseString;
	if ([@[ @"localhost", @"127.0.0.1", @"::1" ] containsObject:host ?: @""]) {
		return YES;
	}
	NSString *served = [[[request valueForHeader:@"host"] ?: @"" componentsSeparatedByString:@":"].firstObject lowercaseString];
	return host.length > 0 && [host isEqualToString:served];
}

/*! The reply to one message, or nil for a notification or a response, which take none. */
+ (nullable NSDictionary *)replyTo:(id)message tools:(NSArray<NFKInferenceTool *> *)tools instructions:(nullable NSString *)instructions
{
	if (![message isKindOfClass:NSDictionary.class] || ![message[@"jsonrpc"] isEqual:@"2.0"]) {
		return [self errorWithIdentifier:nil code:NFKJSONRPCInvalidRequest message:@"not a JSON-RPC 2.0 message"];
	}
	id identifier = message[@"id"];
	NSString *method = [message[@"method"] isKindOfClass:NSString.class] ? message[@"method"] : nil;
	if (identifier == nil || method == nil) {
		return nil;
	}
	NSDictionary *params = [message[@"params"] isKindOfClass:NSDictionary.class] ? message[@"params"] : @{};
	if ([method isEqualToString:@"initialize"]) {
		return [self resultWithIdentifier:identifier result:[self initializeResult:params instructions:instructions]];
	}
	if ([method isEqualToString:@"ping"]) {
		return [self resultWithIdentifier:identifier result:@{}];
	}
	if ([method isEqualToString:@"tools/list"]) {
		NSMutableArray *listed = [NSMutableArray arrayWithCapacity:tools.count];
		for (NFKInferenceTool *tool in tools) {
			[listed addObject:@{ @"name": tool.name, @"description": tool.toolDescription, @"inputSchema": tool.parameters }];
		}
		return [self resultWithIdentifier:identifier result:@{ @"tools": listed }];
	}
	if ([method isEqualToString:@"tools/call"]) {
		NSString *name = [params[@"name"] isKindOfClass:NSString.class] ? params[@"name"] : nil;
		NFKInferenceTool *tool = nil;
		for (NFKInferenceTool *candidate in tools) {
			if ([candidate.name isEqualToString:name]) {
				tool = candidate;
				break;
			}
		}
		if (tool == nil) {
			return [self errorWithIdentifier:identifier code:NFKJSONRPCInvalidParams
									 message:[NSString stringWithFormat:@"no tool named %@", name ?: @"(none)"]];
		}
		NSDictionary *arguments = [params[@"arguments"] isKindOfClass:NSDictionary.class] ? params[@"arguments"] : @{};
		return [self resultWithIdentifier:identifier result:NFKMCPCallResult([tool invokeWithArguments:arguments])];
	}
	return [self errorWithIdentifier:identifier code:NFKJSONRPCMethodNotFound message:[@"no method " stringByAppendingString:method]];
}

// The client's version when the server speaks it, otherwise the newest the server speaks, which a
// client that cannot speak it disconnects from.
+ (NSDictionary *)initializeResult:(NSDictionary *)params instructions:(nullable NSString *)instructions
{
	NSString *requested = [params[@"protocolVersion"] isKindOfClass:NSString.class] ? params[@"protocolVersion"] : nil;
	NSString *version = requested != nil && [NFKMCPProtocolVersions() containsObject:requested] ? requested : NFKMCPProtocolVersions()[0];
	NSMutableDictionary *result = [NSMutableDictionary dictionaryWithDictionary:@{
		@"protocolVersion": version,
		@"capabilities": @{ @"tools": @{ @"listChanged": @NO } },
		@"serverInfo": @{ @"name": @"InferKit", @"version": NFKInferKit.version } }];
	if (instructions.length > 0) {
		result[@"instructions"] = instructions;
	}
	return result;
}

+ (NSDictionary *)resultWithIdentifier:(id)identifier result:(NSDictionary *)result
{
	return @{ @"jsonrpc": @"2.0", @"id": identifier ?: NSNull.null, @"result": result };
}

+ (NSDictionary *)errorWithIdentifier:(nullable id)identifier code:(NSInteger)code message:(NSString *)message
{
	return @{ @"jsonrpc": @"2.0", @"id": identifier ?: NSNull.null, @"error": @{ @"code": @(code), @"message": message } };
}

@end
