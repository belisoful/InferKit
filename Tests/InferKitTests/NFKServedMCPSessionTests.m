//
//  NFKServedMCPSessionTests.m
//  InferKitTests
//
//  MCP sessions on the server's endpoint over real HTTP on a loopback port: initialize issues one,
//  DELETE and idleness end it, the event stream carries pings whose answers keep it, and an unknown or
//  ended session is 404. Intervals are short so each expiry happens within the test.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

/*! Records the session events the server reports. */
@interface NFKSessionRecorder : NSObject <NFKInferenceServerDelegate>
@property (nonatomic, strong) NSMutableArray<NFKServerMCPSession *> *started;
@property (nonatomic, strong) NSMutableArray<NFKServerMCPSession *> *ended;
@end

@implementation NFKSessionRecorder

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_started = [NSMutableArray array];
		_ended = [NSMutableArray array];
	}
	return self;
}

- (void)inferenceServer:(NFKInferenceServer *)server didStartMCPSession:(NFKServerMCPSession *)session
{
	@synchronized (self) {
		[self.started addObject:session];
	}
}

- (void)inferenceServer:(NFKInferenceServer *)server didEndMCPSession:(NFKServerMCPSession *)session
{
	@synchronized (self) {
		[self.ended addObject:session];
	}
}

- (nullable NFKServerMCPSession *)endedSessionWithIdentifier:(NSString *)identifier
{
	@synchronized (self) {
		for (NFKServerMCPSession *session in self.ended) {
			if ([session.identifier isEqualToString:identifier]) {
				return session;
			}
		}
		return nil;
	}
}

@end

/*! Reads a server-sent event stream as it arrives. */
@interface NFKEventStreamReader : NSObject <NSURLSessionDataDelegate>
@property (nonatomic, strong) NSMutableString *text;
@property (atomic, strong, nullable) NSHTTPURLResponse *response;
@property (atomic, assign) BOOL completed;
@end

@implementation NFKEventStreamReader

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_text = [NSMutableString string];
	}
	return self;
}

- (void)URLSession:(NSURLSession *)session dataTask:(NSURLSessionDataTask *)dataTask didReceiveResponse:(NSURLResponse *)response
 completionHandler:(void (^)(NSURLSessionResponseDisposition))completionHandler
{
	self.response = (NSHTTPURLResponse *)response;
	completionHandler(NSURLSessionResponseAllow);
}

- (void)URLSession:(NSURLSession *)session dataTask:(NSURLSessionDataTask *)dataTask didReceiveData:(NSData *)data
{
	@synchronized (self.text) {
		[self.text appendString:[[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] ?: @""];
	}
}

- (void)URLSession:(NSURLSession *)session task:(NSURLSessionTask *)task didCompleteWithError:(nullable NSError *)error
{
	self.completed = YES;
}

/*! The JSON-RPC messages the stream has carried so far. */
- (NSArray<NSDictionary *> *)messages
{
	NSString *text = nil;
	@synchronized (self.text) {
		text = [self.text copy];
	}
	NSMutableArray<NSDictionary *> *messages = [NSMutableArray array];
	for (NSString *line in [text componentsSeparatedByString:@"\n"]) {
		if (![line hasPrefix:@"data: "]) {
			continue;
		}
		id message = [NSJSONSerialization JSONObjectWithData:[[line substringFromIndex:6] dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
		if ([message isKindOfClass:NSDictionary.class]) {
			[messages addObject:message];
		}
	}
	return messages;
}

@end

@interface NFKServedMCPSessionTests : XCTestCase
@property (nonatomic, strong) NFKInferenceServer *server;
@property (nonatomic, strong) NFKSessionRecorder *recorder;
@end

@implementation NFKServedMCPSessionTests

- (void)setUp
{
	[super setUp];
	self.recorder = [[NFKSessionRecorder alloc] init];
	self.server = [[NFKInferenceServer alloc] init];
	self.server.port = 0;
	self.server.loopbackOnly = YES;
	self.server.delegate = self.recorder;
	NSError *error = nil;
	XCTAssertTrue([self.server startWithError:&error], @"%@", error);
}

- (void)tearDown
{
	[self.server stop];
	[super tearDown];
}

#pragma mark Helpers

/*! Sends method to the endpoint with body and headers and returns the parsed reply and the response. */
- (nullable id)send:(NSString *)method body:(nullable id)body headers:(nullable NSDictionary<NSString *, NSString *> *)headers
		   response:(NSHTTPURLResponse **)outResponse
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:self.server.localMCPURL];
	request.HTTPMethod = method;
	if (body != nil) {
		request.HTTPBody = [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL];
		[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	}
	[request setValue:@"application/json, text/event-stream" forHTTPHeaderField:@"Accept"];
	[headers enumerateKeysAndObjectsUsingBlock:^(NSString *field, NSString *value, BOOL *stop) {
		[request setValue:value forHTTPHeaderField:field];
	}];
	NSData *data = [NFKRemoteTransport sendRequest:request session:NSURLSession.sharedSession response:outResponse error:NULL];
	return data.length > 0 ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
}

/*! Initializes a session, presenting key when it is set, and returns the id the reply issued. */
- (NSString *)initializeAs:(NSString *)clientName key:(nullable NSString *)key
{
	NSDictionary *message = @{ @"jsonrpc": @"2.0", @"id": @1, @"method": @"initialize",
							   @"params": @{ @"protocolVersion": @"2025-06-18",
											 @"clientInfo": @{ @"name": clientName, @"version": @"2.1.0" } } };
	NSHTTPURLResponse *response = nil;
	[self send:@"POST" body:message headers:key != nil ? @{ @"Authorization": [@"Bearer " stringByAppendingString:key] } : nil response:&response];
	XCTAssertEqual(response.statusCode, 200);
	NSString *identifier = [response valueForHTTPHeaderField:@"Mcp-Session-Id"];
	XCTAssertGreaterThan(identifier.length, 0, @"initialize issues a session");
	return identifier ?: @"";
}

- (NSInteger)pingStatusInSession:(nullable NSString *)identifier
{
	NSHTTPURLResponse *response = nil;
	[self send:@"POST" body:@{ @"jsonrpc": @"2.0", @"id": @2, @"method": @"ping" }
	   headers:identifier != nil ? @{ @"Mcp-Session-Id": identifier } : nil response:&response];
	return response.statusCode;
}

- (void)waitUntil:(BOOL (^)(void))condition description:(NSString *)description
{
	NSPredicate *predicate = [NSPredicate predicateWithBlock:^BOOL(id object, NSDictionary *bindings) {
		return condition();
	}];
	XCTestExpectation *met = [[XCTNSPredicateExpectation alloc] initWithPredicate:predicate object:nil];
	met.expectationDescription = description;
	[self waitForExpectations:@[ met ] timeout:10];
}

/*! Opens the session's event stream on its own URL session, which the test invalidates at the end. */
- (NFKEventStreamReader *)openStreamForSession:(NSString *)identifier
{
	NFKEventStreamReader *reader = [[NFKEventStreamReader alloc] init];
	NSURLSession *session = [NSURLSession sessionWithConfiguration:NSURLSessionConfiguration.ephemeralSessionConfiguration
														  delegate:reader delegateQueue:nil];
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:self.server.localMCPURL];
	[request setValue:@"text/event-stream" forHTTPHeaderField:@"Accept"];
	[request setValue:identifier forHTTPHeaderField:@"Mcp-Session-Id"];
	[[session dataTaskWithRequest:request] resume];
	[self addTeardownBlock:^{
		[session invalidateAndCancel];
	}];
	return reader;
}

#pragma mark Issuing

- (void)testInitializeIssuesASessionCarryingTheClientsInfo
{
	NSString *identifier = [self initializeAs:@"claude-code" key:nil];

	NSArray<NFKServerMCPSession *> *sessions = self.server.MCPSessions;
	XCTAssertEqual(sessions.count, 1);
	NFKServerMCPSession *session = sessions.firstObject;
	XCTAssertEqualObjects(session.identifier, identifier);
	XCTAssertEqualObjects(session.clientName, @"claude-code");
	XCTAssertEqualObjects(session.clientVersion, @"2.1.0");
	XCTAssertEqualObjects(session.protocolVersion, @"2025-06-18");
	XCTAssertFalse(session.isStreamOpen);
	XCTAssertEqual(session.endReason, NFKServerMCPSessionEndReasonNone);
	XCTAssertNil(session.endDate);
	XCTAssertLessThan(fabs(session.startDate.timeIntervalSinceNow), 5.0);

	XCTAssertEqual([self pingStatusInSession:identifier], 200);
	XCTAssertEqual([self pingStatusInSession:nil], 200, @"a client that never sends the id back is served without a session");
	[self waitUntil:^BOOL {
		@synchronized (self.recorder) {
			return self.recorder.started.count == 1;
		}
	} description:@"the delegate learns of the session"];
	XCTAssertEqualObjects(self.recorder.started.firstObject.clientName, @"claude-code");
}

- (void)testAnUnknownSessionIsNotFound
{
	XCTAssertEqual([self pingStatusInSession:@"no-such-session"], 404);
	NSHTTPURLResponse *response = nil;
	[self send:@"DELETE" body:nil headers:@{ @"Mcp-Session-Id": @"no-such-session" } response:&response];
	XCTAssertEqual(response.statusCode, 404);
	[self send:@"GET" body:nil headers:@{ @"Mcp-Session-Id": @"no-such-session" } response:&response];
	XCTAssertEqual(response.statusCode, 404);
	[self send:@"DELETE" body:nil headers:nil response:&response];
	XCTAssertEqual(response.statusCode, 400, @"DELETE names its session");
}

#pragma mark Ending

- (void)testDeleteEndsTheSession
{
	NSString *identifier = [self initializeAs:@"agent" key:nil];
	NSHTTPURLResponse *response = nil;
	[self send:@"DELETE" body:nil headers:@{ @"Mcp-Session-Id": identifier } response:&response];
	XCTAssertEqual(response.statusCode, 204);
	XCTAssertEqual(self.server.MCPSessions.count, 0);
	XCTAssertEqual([self pingStatusInSession:identifier], 404, @"an ended session's id is not found");

	[self waitUntil:^BOOL {
		return [self.recorder endedSessionWithIdentifier:identifier] != nil;
	} description:@"the delegate learns the session ended"];
	NFKServerMCPSession *ended = [self.recorder endedSessionWithIdentifier:identifier];
	XCTAssertEqual(ended.endReason, NFKServerMCPSessionEndReasonDeleted);
	XCTAssertNotNil(ended.endDate);
}

- (void)testAnIdleSessionEnds
{
	self.server.MCPSessionIdleTimeout = 0.3;
	NSString *identifier = [self initializeAs:@"agent" key:nil];
	XCTAssertEqual([self pingStatusInSession:identifier], 200, @"activity within the timeout keeps the session");
	[self waitUntil:^BOOL {
		return self.server.MCPSessions.count == 0;
	} description:@"the idle session ends"];
	XCTAssertEqual([self pingStatusInSession:identifier], 404);
	[self waitUntil:^BOOL {
		return [self.recorder endedSessionWithIdentifier:identifier] != nil;
	} description:@"the delegate learns the session ended"];
	XCTAssertEqual([self.recorder endedSessionWithIdentifier:identifier].endReason, NFKServerMCPSessionEndReasonIdle);
}

- (void)testTheHostAndAStoppingServerEndSessions
{
	NSString *first = [self initializeAs:@"first" key:nil];
	NSString *second = [self initializeAs:@"second" key:nil];
	XCTAssertTrue([self.server endMCPSessionWithIdentifier:first]);
	XCTAssertFalse([self.server endMCPSessionWithIdentifier:first]);
	XCTAssertEqual([self pingStatusInSession:first], 404);

	[self.server stop];
	XCTAssertEqual(self.server.MCPSessions.count, 0);
	[self waitUntil:^BOOL {
		return [self.recorder endedSessionWithIdentifier:second] != nil;
	} description:@"the stop ends the second session"];
	XCTAssertEqual([self.recorder endedSessionWithIdentifier:first].endReason, NFKServerMCPSessionEndReasonEndedByHost);
	XCTAssertEqual([self.recorder endedSessionWithIdentifier:second].endReason, NFKServerMCPSessionEndReasonServerStopped);
}

#pragma mark The stream

- (void)testAnsweredPingsKeepASessionAndMissedOnesEndIt
{
	self.server.MCPPingInterval = 0.1;
	self.server.MCPMissedPingLimit = 2;
	self.server.MCPSessionIdleTimeout = 0.2;
	NSString *identifier = [self initializeAs:@"agent" key:nil];
	NFKEventStreamReader *reader = [self openStreamForSession:identifier];
	[self waitUntil:^BOOL {
		return self.server.MCPSessions.firstObject.isStreamOpen;
	} description:@"the stream opens"];
	XCTAssertEqual(reader.response.statusCode, 200);
	XCTAssertEqualObjects([reader.response valueForHTTPHeaderField:@"Content-Type"], @"text/event-stream");

	NSMutableSet *answered = [NSMutableSet set];
	NSDate *until = [NSDate dateWithTimeIntervalSinceNow:1.0];
	while (until.timeIntervalSinceNow > 0) {
		for (NSDictionary *message in [reader messages]) {
			if ([message[@"method"] isEqual:@"ping"] && ![answered containsObject:message[@"id"]]) {
				[answered addObject:message[@"id"]];
				NSHTTPURLResponse *response = nil;
				[self send:@"POST" body:@{ @"jsonrpc": @"2.0", @"id": message[@"id"], @"result": @{} }
				   headers:@{ @"Mcp-Session-Id": identifier } response:&response];
				XCTAssertEqual(response.statusCode, 202, @"an answer is acknowledged");
			}
		}
		[NSThread sleepForTimeInterval:0.02];
	}
	XCTAssertGreaterThanOrEqual(answered.count, 3);
	XCTAssertEqual(self.server.MCPSessions.count, 1, @"answered pings keep the session past its idle timeout");

	[self waitUntil:^BOOL {
		return self.server.MCPSessions.count == 0;
	} description:@"unanswered pings end the session"];
	[self waitUntil:^BOOL {
		return reader.completed;
	} description:@"the session's end closes its stream"];
	[self waitUntil:^BOOL {
		return [self.recorder endedSessionWithIdentifier:identifier] != nil;
	} description:@"the delegate learns the session ended"];
	XCTAssertEqual([self.recorder endedSessionWithIdentifier:identifier].endReason, NFKServerMCPSessionEndReasonPingTimeout);
}

#pragma mark Keys

- (void)testASessionBelongsToItsKeyAndEndsWhenTheKeyIsRevoked
{
	self.server.requiresAPIKeyOnLoopback = YES;
	[self.server addAPIKey:@"key-one" named:@"one" scopes:NFKServerKeyScopeMCP];
	[self.server addAPIKey:@"key-two" named:@"two" scopes:NFKServerKeyScopeMCP];
	NSString *identifier = [self initializeAs:@"agent" key:@"key-one"];
	XCTAssertEqualObjects(self.server.MCPSessions.firstObject.keyName, @"one");

	NSHTTPURLResponse *response = nil;
	NSDictionary *ping = @{ @"jsonrpc": @"2.0", @"id": @2, @"method": @"ping" };
	[self send:@"POST" body:ping headers:@{ @"Mcp-Session-Id": identifier, @"Authorization": @"Bearer key-two" } response:&response];
	XCTAssertEqual(response.statusCode, 404, @"another key cannot use the session");
	[self send:@"POST" body:ping headers:@{ @"Mcp-Session-Id": identifier, @"Authorization": @"Bearer key-one" } response:&response];
	XCTAssertEqual(response.statusCode, 200);

	[self.server removeAPIKeyNamed:@"one"];
	XCTAssertEqual(self.server.MCPSessions.count, 0);
	[self waitUntil:^BOOL {
		return [self.recorder endedSessionWithIdentifier:identifier] != nil;
	} description:@"the delegate learns the session ended"];
	XCTAssertEqual([self.recorder endedSessionWithIdentifier:identifier].endReason, NFKServerMCPSessionEndReasonRevoked);
}

@end
