//
//  NFKServedMCP.m
//  InferKit
//

#import "NFKServedMCP.h"
#import "NFKServerObservationPrivate.h"
#import "NFKServedRequestChecks.h"
#import <InferKit/NFKInferKit.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKAudioAsset.h>
#import <InferKit/NFKVideoAsset.h>

static NSInteger const NFKJSONRPCParseError		= -32700;
static NSInteger const NFKJSONRPCInvalidRequest	= -32600;
static NSInteger const NFKJSONRPCMethodNotFound	= -32601;
static NSInteger const NFKJSONRPCInvalidParams	= -32602;
static NSInteger const NFKJSONRPCInternalError	= -32603;

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

/*! The seconds an issued session waits for its client to send the id back, unless idleTimeout is shorter. */
static NSTimeInterval const NFKMCPPendingWindow = 60;

/*! The issued sessions that wait at once; issuing one more discards the oldest. */
static NSUInteger const NFKMCPPendingLimit = 256;

static NSTimeInterval NFKMCPNow(void)
{
	return NSProcessInfo.processInfo.systemUptime;
}

static BOOL NFKMCPSameKeyName(NSString * _Nullable left, NSString * _Nullable right)
{
	return left == right || [left isEqualToString:right];
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
/*! When initialize issued the session, then when the client's first request carrying its id started it. */
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
/*! The active sessions by identifier. */
@property (nonatomic, strong) NSMutableDictionary<NSString *, NFKServedMCPSessionRecord *> *records;
/*! The issued sessions whose clients have not yet sent the id back, by identifier. */
@property (nonatomic, strong) NSMutableDictionary<NSString *, NFKServedMCPSessionRecord *> *pendingRecords;
@property (nonatomic, strong) dispatch_queue_t queue;
/*! Delivers the event handler's calls in order. */
@property (nonatomic, strong) dispatch_queue_t eventQueue;
@property (nonatomic, strong, nullable) dispatch_source_t timer;
/*! The period the timer runs at, or 0 while it is stopped. */
@property (nonatomic, assign) NSTimeInterval timerInterval;
/*! The monotonic clock in seconds that every activity and expiry reads, which a test replaces. */
@property (atomic, copy) NSTimeInterval (^uptimeProvider)(void);
@end

@implementation NFKServedMCPSessionStore

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_records = [NSMutableDictionary dictionary];
		_pendingRecords = [NSMutableDictionary dictionary];
		_idleTimeout = 1800;
		_pingInterval = 30;
		_missedPingLimit = 3;
		_queue = dispatch_queue_create("NFKInferenceServer.mcp", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0));
		_eventQueue = dispatch_queue_create("NFKInferenceServer.mcp.events", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0));
		_uptimeProvider = ^NSTimeInterval {
			return NFKMCPNow();
		};
	}
	return self;
}

- (void)dealloc
{
	if (_timer != nil) {
		dispatch_source_cancel(_timer);
	}
}

- (NSTimeInterval)uptime
{
	return self.uptimeProvider();
}

/*! Runs on the lock, so the handler receives the events in the order they happened. */
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

/*! Issues a session for an initialize and returns its identifier. The session waits, unlisted and
	unreported, for the client's first request that carries the identifier. */
- (NSString *)issueSessionForRequest:(NFKHTTPRequest *)request
						  clientInfo:(nullable NSDictionary *)clientInfo
					 protocolVersion:(NSString *)protocolVersion
{
	NFKServedMCPSessionRecord *record = [[NFKServedMCPSessionRecord alloc] init];
	record.identifier = NSUUID.UUID.UUIDString;
	record.clientName = [clientInfo[@"name"] isKindOfClass:NSString.class] ? clientInfo[@"name"] : nil;
	record.clientVersion = [clientInfo[@"version"] isKindOfClass:NSString.class] ? clientInfo[@"version"] : nil;
	record.protocolVersion = protocolVersion;
	record.clientAddress = request.clientAddress ?: @"";
	record.keyName = request.keyName;
	record.startDate = [NSDate date];
	@synchronized (self) {
		record.startedAt = [self uptime];
		record.lastActivity = record.startedAt;
		[self makeRoomForPendingSession];
		self.pendingRecords[record.identifier] = record;
		[self updateTimer];
	}
	return record.identifier;
}

/*! Discards the oldest issued sessions until one more fits. Runs on the lock. */
- (void)makeRoomForPendingSession
{
	while (self.pendingRecords.count >= NFKMCPPendingLimit) {
		NFKServedMCPSessionRecord *oldest = nil;
		for (NFKServedMCPSessionRecord *record in self.pendingRecords.allValues) {
			if (oldest == nil || record.startedAt < oldest.startedAt) {
				oldest = record;
			}
		}
		if (oldest == nil) {
			break;
		}
		[self.pendingRecords removeObjectForKey:oldest.identifier];
	}
}

/*! The session a request names when it started with the same key name, or nil. A request that names an
	issued session starts it: the session joins sessions and the event handler learns of it. */
- (nullable NFKServedMCPSessionRecord *)sessionWithIdentifier:(NSString *)identifier keyName:(nullable NSString *)keyName
{
	@synchronized (self) {
		NFKServedMCPSessionRecord *record = self.records[identifier];
		if (record == nil) {
			return [self startPendingSessionWithIdentifier:identifier keyName:keyName];
		}
		return NFKMCPSameKeyName(record.keyName, keyName) ? record : nil;
	}
}

/*! Runs on the lock. */
- (nullable NFKServedMCPSessionRecord *)startPendingSessionWithIdentifier:(NSString *)identifier keyName:(nullable NSString *)keyName
{
	NFKServedMCPSessionRecord *record = self.pendingRecords[identifier];
	if (record == nil || !NFKMCPSameKeyName(record.keyName, keyName)) {
		return nil;
	}
	NSTimeInterval now = [self uptime];
	[self.pendingRecords removeObjectForKey:identifier];
	record.startDate = [NSDate date];
	record.startedAt = now;
	record.lastActivity = now;
	self.records[identifier] = record;
	[self report:[record snapshotAt:now endReason:NFKServerMCPSessionEndReasonNone] ended:NO];
	[self updateTimer];
	return record;
}

/*! Ends the active session a DELETE names, or discards the issued session it names without reporting
	it; NO when it names neither under the same key name. */
- (BOOL)deleteSessionWithIdentifier:(NSString *)identifier keyName:(nullable NSString *)keyName
{
	NFKServedMCPSessionRecord *record = nil;
	@synchronized (self) {
		NFKServedMCPSessionRecord *pending = self.pendingRecords[identifier];
		if (pending != nil && NFKMCPSameKeyName(pending.keyName, keyName)) {
			[self.pendingRecords removeObjectForKey:identifier];
			[self updateTimer];
			return YES;
		}
		record = self.records[identifier];
		if (record == nil || !NFKMCPSameKeyName(record.keyName, keyName)) {
			return NO;
		}
	}
	return [self endSession:record reason:NFKServerMCPSessionEndReasonDeleted];
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
	@synchronized (self) {
		if (self.records[record.identifier] != record) {
			return NO;
		}
		[self.records removeObjectForKey:record.identifier];
		[self report:[record snapshotAt:[self uptime] endReason:reason] ended:YES];
		stream = record.stream;
		record.stream = nil;
		[self updateTimer];
	}
	[stream endStream];
	return YES;
}

- (void)endSessionsWithKeyName:(NSString *)keyName reason:(NFKServerMCPSessionEndReason)reason
{
	NSArray<NFKServedMCPSessionRecord *> *matching = nil;
	@synchronized (self) {
		matching = [self.records.allValues filteredArrayUsingPredicate:[NSPredicate predicateWithBlock:^BOOL(NFKServedMCPSessionRecord *record, NSDictionary *bindings) {
			return [record.keyName isEqualToString:keyName];
		}]];
		NSSet<NSString *> *pending = [self.pendingRecords keysOfEntriesPassingTest:^BOOL(NSString *identifier, NFKServedMCPSessionRecord *record, BOOL *stop) {
			return [record.keyName isEqualToString:keyName];
		}];
		[self.pendingRecords removeObjectsForKeys:pending.allObjects];
		[self updateTimer];
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
		[self.pendingRecords removeAllObjects];
		[self updateTimer];
	}
	for (NFKServedMCPSessionRecord *record in all) {
		[self endSession:record reason:reason];
	}
}

- (NSArray<NFKServerMCPSession *> *)sessions
{
	@synchronized (self) {
		NSTimeInterval now = [self uptime];
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
		record.lastActivity = [self uptime];
	}
}

- (void)endRequestInSession:(nullable NFKServedMCPSessionRecord *)record
{
	@synchronized (self) {
		record.inFlight -= record.inFlight > 0 ? 1 : 0;
		record.lastActivity = [self uptime];
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
		record.lastActivity = [self uptime];
	}
}

#pragma mark Streams

/*! Makes response the session's event stream, closing the one it replaces; NO when the session has
	ended. */
- (BOOL)openStream:(NFKHTTPResponse *)response forSession:(NFKServedMCPSessionRecord *)record
{
	__weak NFKServedMCPSessionStore *weakSelf = self;
	__weak NFKServedMCPSessionRecord *weakRecord = record;
	__weak NFKHTTPResponse *weakResponse = response;
	response.disconnectHandler = ^{
		[weakSelf streamClosed:weakResponse inSession:weakRecord];
	};
	NFKHTTPResponse *replaced = nil;
	@synchronized (self) {
		if (self.records[record.identifier] != record) {
			return NO;
		}
		replaced = record.stream;
		record.stream = response;
		record.outstandingPings = 0;
		record.lastPingAt = [self uptime];
		record.lastActivity = record.lastPingAt;
	}
	[response beginStreamWithStatus:200 headers:@{ @"Content-Type": @"text/event-stream", @"Mcp-Session-Id": record.identifier }];
	// NSURLSession reports a streamed response only once body bytes arrive, so an SSE comment, which
	// clients ignore, goes out at once.
	[response sendStreamData:[@": open\n\n" dataUsingEncoding:NSUTF8StringEncoding]];
	[replaced endStream];
	// A client that left before the handler was installed never fires it.
	if (response.disconnected) {
		[self streamClosed:response inSession:record];
	}
	return YES;
}

- (void)streamClosed:(nullable NFKHTTPResponse *)response inSession:(nullable NFKServedMCPSessionRecord *)record
{
	@synchronized (self) {
		if (response != nil && record.stream == response) {
			record.stream = nil;
			record.outstandingPings = 0;
			record.lastActivity = [self uptime];
		}
	}
}

#pragma mark Expiry

- (void)settingsChanged
{
	@synchronized (self) {
		[self updateTimer];
	}
}

/*! The seconds an issued session waits for its client: NFKMCPPendingWindow, or idleTimeout when shorter. */
- (NSTimeInterval)pendingWindow
{
	NSTimeInterval idleTimeout = self.idleTimeout;
	return idleTimeout > 0 ? MIN(idleTimeout, NFKMCPPendingWindow) : NFKMCPPendingWindow;
}

/*! A quarter of the shortest interval the held sessions need watched, within 10 ms and 1 s, or 0 when
	none needs watching. Runs on the lock. */
- (NSTimeInterval)tickInterval
{
	NSTimeInterval shortest = INFINITY;
	if (self.records.count > 0 && self.pingInterval > 0) {
		shortest = MIN(shortest, self.pingInterval);
	}
	if (self.records.count > 0 && self.idleTimeout > 0) {
		shortest = MIN(shortest, self.idleTimeout);
	}
	if (self.pendingRecords.count > 0) {
		shortest = MIN(shortest, [self pendingWindow]);
	}
	return isfinite(shortest) ? MIN(MAX(shortest / 4.0, 0.01), 1.0) : 0;
}

/*! Runs the timer at the tick interval the held sessions and the settings call for, or stops it. Runs on
	the lock. */
- (void)updateTimer
{
	NSTimeInterval tick = [self tickInterval];
	if (self.timer != nil && tick == self.timerInterval) {
		return;
	}
	[self cancelTimer];
	if (tick <= 0) {
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
	self.timerInterval = tick;
}

/*! Runs on the lock. */
- (void)cancelTimer
{
	if (self.timer != nil) {
		dispatch_source_cancel(self.timer);
		self.timer = nil;
	}
	self.timerInterval = 0;
}

/*! Discards the issued sessions whose clients never sent the id back, pings each open stream that is due
	one, and ends the sessions that missed too many pings or sat idle. */
- (void)sweep
{
	NSTimeInterval pingInterval = self.pingInterval;
	NSTimeInterval idleTimeout = self.idleTimeout;
	NSTimeInterval pendingWindow = [self pendingWindow];
	NSUInteger missedLimit = MAX(self.missedPingLimit, (NSUInteger)1);
	NSMutableArray<NFKServedMCPSessionRecord *> *missed = [NSMutableArray array];
	NSMutableArray<NFKServedMCPSessionRecord *> *idle = [NSMutableArray array];
	NSMutableArray<NSArray *> *pings = [NSMutableArray array];
	@synchronized (self) {
		NSTimeInterval now = [self uptime];
		NSSet<NSString *> *unclaimed = [self.pendingRecords keysOfEntriesPassingTest:^BOOL(NSString *identifier, NFKServedMCPSessionRecord *record, BOOL *stop) {
			return now - record.startedAt >= pendingWindow;
		}];
		[self.pendingRecords removeObjectsForKeys:unclaimed.allObjects];
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
		[self updateTimer];
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

/*! The replies of one POST in message order, some given at once and some filled when a started tool call
	finishes, and the cancel blocks of those calls, which run when the client leaves. */
@interface NFKServedMCPExchange : NSObject
- (instancetype)initWithResponse:(NFKHTTPResponse *)response;
/*! Appends a reply; malformed marks one that refuses a message of the wrong type. */
- (void)addReply:(NSDictionary *)reply malformed:(BOOL)malformed;
/*! Appends a slot that fillSlot:reply: fills once, and returns its index. */
- (NSUInteger)reserveSlot;
- (void)fillSlot:(NSUInteger)slot reply:(NSDictionary *)reply;
- (void)addCancel:(dispatch_block_t)cancel;
/*! Calls handler on a global queue once every slot is filled. */
- (void)finishWithHandler:(void (^)(NSArray<NSDictionary *> *replies, BOOL malformed))handler;
@end

@interface NFKServedMCPExchange ()
@property (nonatomic, strong) NSMutableArray *replies;
@property (nonatomic, strong) NSMutableArray<dispatch_block_t> *cancels;
@property (nonatomic, strong) dispatch_group_t group;
@property (nonatomic, assign) BOOL malformed;
/*! Whether the client left, after which a call that starts is cancelled at once. */
@property (nonatomic, assign) BOOL clientLeft;
@end

@implementation NFKServedMCPExchange

- (instancetype)initWithResponse:(NFKHTTPResponse *)response
{
	self = [super init];
	if (self != nil) {
		_replies = [NSMutableArray array];
		_cancels = [NSMutableArray array];
		_group = dispatch_group_create();
		__weak NFKServedMCPExchange *weakSelf = self;
		response.disconnectHandler = ^{
			[weakSelf cancelCalls];
		};
	}
	return self;
}

- (void)addReply:(NSDictionary *)reply malformed:(BOOL)malformed
{
	@synchronized (self) {
		[self.replies addObject:reply];
		self.malformed = self.malformed || malformed;
	}
}

- (NSUInteger)reserveSlot
{
	dispatch_group_enter(self.group);
	@synchronized (self) {
		[self.replies addObject:NSNull.null];
		return self.replies.count - 1;
	}
}

- (void)fillSlot:(NSUInteger)slot reply:(NSDictionary *)reply
{
	@synchronized (self) {
		if (self.replies[slot] != NSNull.null) {
			return;
		}
		self.replies[slot] = reply;
	}
	dispatch_group_leave(self.group);
}

- (void)addCancel:(dispatch_block_t)cancel
{
	BOOL cancelNow = NO;
	@synchronized (self) {
		[self.cancels addObject:cancel];
		cancelNow = self.clientLeft;
	}
	if (cancelNow) {
		cancel();
	}
}

- (void)cancelCalls
{
	NSArray<dispatch_block_t> *cancels = nil;
	@synchronized (self) {
		self.clientLeft = YES;
		cancels = [self.cancels copy];
	}
	for (dispatch_block_t cancel in cancels) {
		cancel();
	}
}

- (void)finishWithHandler:(void (^)(NSArray<NSDictionary *> *replies, BOOL malformed))handler
{
	dispatch_group_notify(self.group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NSArray<NSDictionary *> *replies = nil;
		BOOL malformed = NO;
		@synchronized (self) {
			replies = [self.replies copy];
			malformed = self.malformed;
		}
		handler(replies, malformed);
	});
}

@end

@implementation NFKServedMCP

+ (void)serveRequest:(NFKHTTPRequest *)request
			response:(NFKHTTPResponse *)response
			   tools:(NSArray<NFKInferenceTool *> *)tools
		instructions:(nullable NSString *)instructions
			sessions:(NFKServedMCPSessionStore *)sessions
		 toolStarter:(nullable NFKServedMCPToolStarter)toolStarter
{
	if (!NFKServedOriginIsAllowed(request)) {
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
	if (!NFKServedContentTypeAcceptsJSON([request valueForHeader:@"content-type"])) {
		[response sendStatus:415 JSONObject:[self errorWithIdentifier:nil code:NFKJSONRPCInvalidRequest
															  message:@"the body is JSON-RPC; send it with Content-Type: application/json"] headers:nil];
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
		session = [sessions sessionWithIdentifier:sessionIdentifier keyName:request.keyName];
		if (session == nil) {
			[self sendUnknownSession:response];
			return;
		}
	}
	[sessions beginRequestInSession:session];
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NFKServedMCPExchange *exchange = [[NFKServedMCPExchange alloc] initWithResponse:response];
		NSString *issued = nil;
		for (id message in messages) {
			if ([self isResponse:message]) {
				[sessions receivedResponseWithIdentifier:message[@"id"] inSession:session];
				continue;
			}
			NSDictionary *reply = [self answer:message tools:tools instructions:instructions toolStarter:toolStarter exchange:exchange];
			NSDictionary *result = reply[@"result"];
			if (issued == nil && [message[@"method"] isEqual:@"initialize"] && [result isKindOfClass:NSDictionary.class]) {
				NSDictionary *params = [message[@"params"] isKindOfClass:NSDictionary.class] ? message[@"params"] : @{};
				NSDictionary *clientInfo = [params[@"clientInfo"] isKindOfClass:NSDictionary.class] ? params[@"clientInfo"] : nil;
				issued = [sessions issueSessionForRequest:request clientInfo:clientInfo protocolVersion:result[@"protocolVersion"]];
			}
		}
		NSDictionary *headers = issued != nil ? @{ @"Mcp-Session-Id": issued } : nil;
		[exchange finishWithHandler:^(NSArray<NSDictionary *> *replies, BOOL malformed) {
			[sessions endRequestInSession:session];
			if (replies.count == 0) {
				[response sendStatus:202 headers:headers body:nil];
				return;
			}
			[response sendStatus:!batch && malformed ? 400 : 200 JSONObject:batch ? replies : replies.firstObject headers:headers];
		}];
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
	if (!opensStream) {
		if ([sessions deleteSessionWithIdentifier:identifier keyName:request.keyName]) {
			[response sendStatus:204 headers:nil body:nil];
		} else {
			[self sendUnknownSession:response];
		}
		return;
	}
	NFKServedMCPSessionRecord *session = [sessions sessionWithIdentifier:identifier keyName:request.keyName];
	if (session == nil || ![sessions openStream:response forSession:session]) {
		[self sendUnknownSession:response];
	}
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

/*! The refusal of a message whose envelope is not JSON-RPC 2.0: not an object, another jsonrpc, an id
	that is not a string, a number, or null, or a method that is not a string. nil for a well-formed one. */
+ (nullable NSDictionary *)envelopeErrorOf:(id)message
{
	if (![message isKindOfClass:NSDictionary.class] || ![message[@"jsonrpc"] isEqual:@"2.0"]) {
		return [self errorWithIdentifier:nil code:NFKJSONRPCInvalidRequest message:@"not a JSON-RPC 2.0 message"];
	}
	id identifier = message[@"id"];
	if (identifier != nil && !NFKServedValueIsOfType(identifier, @"string|number|null")) {
		return [self errorWithIdentifier:nil code:NFKJSONRPCInvalidRequest message:@"id must be a string, a number, or null"];
	}
	NSError *typeError = NFKServedFieldTypeError(message, @{ @"method": @"string" }, nil);
	return typeError != nil ? [self errorWithIdentifier:identifier code:NFKJSONRPCInvalidRequest message:typeError.localizedDescription] : nil;
}

/*! The refusal of a request's params whose fields are of the wrong type for its method, or nil. */
+ (nullable NSError *)paramsTypeErrorOf:(id)params method:(NSString *)method
{
	NSDictionary<NSString *, NSDictionary<NSString *, NSString *> *> *fields = @{
		@"initialize": @{ @"protocolVersion": @"string", @"capabilities": @"object", @"clientInfo": @"object" },
		@"tools/call": @{ @"name": @"string", @"arguments": @"object" } };
	return NFKServedFieldTypeError(params, fields[method] ?: @{}, @"params");
}

/*! Answers one message into exchange: a reply now, a slot a started tool call fills later, or nothing for
	a notification. Returns the reply it gave now, which initialize's session reads. */
+ (nullable NSDictionary *)answer:(id)message
							tools:(NSArray<NFKInferenceTool *> *)tools
					 instructions:(nullable NSString *)instructions
					  toolStarter:(nullable NFKServedMCPToolStarter)toolStarter
						 exchange:(NFKServedMCPExchange *)exchange
{
	NSDictionary *refusal = [self envelopeErrorOf:message];
	if (refusal != nil) {
		[exchange addReply:refusal malformed:YES];
		return refusal;
	}
	id identifier = message[@"id"];
	NSString *method = message[@"method"];
	if (identifier != nil && method == nil) {
		NSDictionary *reply = [self errorWithIdentifier:identifier code:NFKJSONRPCInvalidRequest message:@"a request names its method"];
		[exchange addReply:reply malformed:YES];
		return reply;
	}
	if (identifier == nil) {
		return nil;
	}
	id params = message[@"params"];
	NSError *typeError = [self paramsTypeErrorOf:params method:method];
	if (typeError != nil) {
		NSDictionary *reply = [self errorWithIdentifier:identifier code:NFKJSONRPCInvalidParams message:typeError.localizedDescription];
		[exchange addReply:reply malformed:YES];
		return reply;
	}
	params = [params isKindOfClass:NSDictionary.class] ? NFKServedWithoutNulls(params) : @{};
	NFKInferenceTool *tool = [method isEqualToString:@"tools/call"] ? [self toolNamed:params[@"name"] in:tools] : nil;
	if (tool != nil && toolStarter != nil) {
		NSUInteger slot = [exchange reserveSlot];
		__block BOOL completed = NO;
		dispatch_block_t cancel = nil;
		@try {
			cancel = toolStarter(tool, params[@"arguments"] ?: @{}, ^(NFKToolResult *result) {
				completed = YES;
				[exchange fillSlot:slot reply:[self callReplyWithIdentifier:identifier result:result]];
			});
		} @catch (NSException *exception) {
			[exchange fillSlot:slot reply:[self errorWithIdentifier:identifier code:NFKJSONRPCInternalError
														   message:NFKServedExceptionError(exception).localizedDescription]];
			return nil;
		}
		if (cancel != nil) {
			[exchange addCancel:cancel];
			return nil;
		}
		if (completed) {
			return nil;
		}
		[exchange fillSlot:slot reply:[self replyTo:message params:params tools:tools instructions:instructions]];
		return nil;
	}
	NSDictionary *reply = [self replyTo:message params:params tools:tools instructions:instructions];
	[exchange addReply:reply malformed:NO];
	return reply;
}

+ (nullable NFKInferenceTool *)toolNamed:(nullable NSString *)name in:(NSArray<NFKInferenceTool *> *)tools
{
	for (NFKInferenceTool *candidate in tools) {
		if ([candidate.name isEqualToString:name]) {
			return candidate;
		}
	}
	return nil;
}

/*! A tools/call reply for a result, or -32603 when the result cannot be written as MCP content. */
+ (NSDictionary *)callReplyWithIdentifier:(id)identifier result:(NFKToolResult *)result
{
	@try {
		return [self resultWithIdentifier:identifier result:NFKMCPCallResult(result)];
	} @catch (NSException *exception) {
		return [self errorWithIdentifier:identifier code:NFKJSONRPCInternalError message:NFKServedExceptionError(exception).localizedDescription];
	}
}

/*! The reply to a well-formed request, run on the calling thread; a tool that raises answers -32603. */
+ (NSDictionary *)replyTo:(NSDictionary *)message params:(NSDictionary *)params tools:(NSArray<NFKInferenceTool *> *)tools instructions:(nullable NSString *)instructions
{
	id identifier = message[@"id"];
	NSString *method = message[@"method"];
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
		NSString *name = params[@"name"];
		NFKInferenceTool *tool = [self toolNamed:name in:tools];
		if (tool == nil) {
			return [self errorWithIdentifier:identifier code:NFKJSONRPCInvalidParams
									 message:[NSString stringWithFormat:@"no tool named %@", name ?: @"(none)"]];
		}
		@try {
			return [self callReplyWithIdentifier:identifier result:[tool invokeWithArguments:params[@"arguments"] ?: @{}]];
		} @catch (NSException *exception) {
			return [self errorWithIdentifier:identifier code:NFKJSONRPCInternalError message:NFKServedExceptionError(exception).localizedDescription];
		}
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
		@"serverInfo": @{ @"name": @"InferKit", @"version": NFKInferKit.releaseVersion } }];
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
