//
//  NFKServerRouteTests.m
//  InferKitTests
//
//  An app's own routes on the server's listener, over real HTTP on a loopback port: what a handler
//  receives, whole and streamed replies, prefix matching and refusal, and the limits and key checks
//  the server's own routes keep.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

@interface NFKServerRouteTests : XCTestCase
@property (nonatomic, strong) NFKInferenceServer *server;
@end

@implementation NFKServerRouteTests

- (void)setUp
{
	[super setUp];
	self.server = [[NFKInferenceServer alloc] init];
	self.server.port = 0;
	self.server.loopbackOnly = YES;
	NSError *error = nil;
	XCTAssertTrue([self.server startWithError:&error], @"%@", error);
}

- (void)tearDown
{
	[self.server stop];
	[super tearDown];
}

#pragma mark Helpers

- (NSURL *)URLForPath:(NSString *)path
{
	return [NSURL URLWithString:[NSString stringWithFormat:@"http://localhost:%u%@", (unsigned)self.server.listeningPort, path]];
}

/*! Sends method to path with body and headers, and returns the reply's body, its status, and its head. */
- (nullable NSData *)send:(NSString *)method to:(NSString *)path body:(nullable NSData *)body
				  headers:(nullable NSDictionary<NSString *, NSString *> *)headers response:(NSHTTPURLResponse **)outResponse
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:[self URLForPath:path]];
	request.HTTPMethod = method;
	request.HTTPBody = body;
	[headers enumerateKeysAndObjectsUsingBlock:^(NSString *field, NSString *value, BOOL *stop) {
		[request setValue:value forHTTPHeaderField:field];
	}];
	return [NFKRemoteTransport sendRequest:request session:NSURLSession.sharedSession response:outResponse error:NULL];
}

/*! A handler that answers with what it received as JSON, under name. */
- (NFKServerRouteHandler)echoHandlerNamed:(NSString *)name
{
	return ^(NFKServerRouteRequest *request, NFKServerRouteResponse *response) {
		[response sendStatus:200 JSONObject:@{ @"handler": name,
											   @"method": request.method,
											   @"path": request.path,
											   @"prefix": request.pathPrefix,
											   @"query": request.query,
											   @"header": [request valueForHeader:@"X-Probe"] ?: @"",
											   @"body": [[NSString alloc] initWithData:request.body encoding:NSUTF8StringEncoding] ?: @"",
											   @"loopback": @(request.fromLoopback),
											   @"address": request.clientAddress,
											   @"main": @(NSThread.isMainThread) }
					 headers:nil];
	};
}

- (nullable NSDictionary *)JSONFrom:(nullable NSData *)data
{
	return data.length > 0 ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
}

#pragma mark Requests and replies

- (void)testAHandlerReceivesTheRequestUnderItsPrefix
{
	NSError *error = nil;
	XCTAssertTrue([self.server addRouteForPathPrefix:@"/myapp" handler:[self echoHandlerNamed:@"app"] error:&error], @"%@", error);
	XCTAssertEqualObjects(self.server.routePathPrefixes, @[ @"/myapp/" ]);

	NSHTTPURLResponse *response = nil;
	NSDictionary *reply = [self JSONFrom:[self send:@"GET" to:@"/myapp/items/7?sort=name&q=a%20b" body:nil
											headers:@{ @"X-Probe": @"probe" } response:&response]];
	XCTAssertEqual(response.statusCode, 200);
	XCTAssertEqualObjects(reply[@"method"], @"GET");
	XCTAssertEqualObjects(reply[@"path"], @"/myapp/items/7");
	XCTAssertEqualObjects(reply[@"prefix"], @"/myapp/");
	XCTAssertEqualObjects(reply[@"query"], (@{ @"sort": @"name", @"q": @"a b" }));
	XCTAssertEqualObjects(reply[@"header"], @"probe");
	XCTAssertEqualObjects(reply[@"loopback"], @YES);
	NSArray<NSString *> *loopbackAddresses = @[ @"127.0.0.1", @"::1" ];
	XCTAssertTrue([loopbackAddresses containsObject:reply[@"address"]], @"%@", reply[@"address"]);
	XCTAssertEqualObjects(reply[@"main"], @NO, @"a handler runs off the main thread");

	reply = [self JSONFrom:[self send:@"GET" to:@"/myapp" body:nil headers:nil response:&response]];
	XCTAssertEqual(response.statusCode, 200, @"the prefix itself matches");
	XCTAssertEqualObjects(reply[@"path"], @"/myapp");
}

- (void)testAHandlerAnswersWithItsOwnStatusHeadersAndBodyLater
{
	[self.server addRouteForPathPrefix:@"/myapp/" handler:^(NFKServerRouteRequest *request, NFKServerRouteResponse *response) {
		NSData *body = request.body;
		dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.1 * NSEC_PER_SEC)), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			[response sendStatus:201 headers:@{ @"Content-Type": @"application/octet-stream", @"X-Echo": request.method } body:body];
		});
	} error:NULL];

	NSData *payload = [@"twelve bytes" dataUsingEncoding:NSUTF8StringEncoding];
	NSHTTPURLResponse *response = nil;
	NSData *reply = [self send:@"PUT" to:@"/myapp/blob" body:payload headers:nil response:&response];
	XCTAssertEqual(response.statusCode, 201);
	XCTAssertEqualObjects([response valueForHTTPHeaderField:@"X-Echo"], @"PUT");
	XCTAssertEqualObjects(reply, payload);
}

- (void)testAFinishedReplyReleasesADisconnectHandlerThatCapturesIt
{
	__block __weak NFKServerRouteResponse *captured = nil;
	[self.server addRouteForPathPrefix:@"/myapp/" handler:^(NFKServerRouteRequest *request, NFKServerRouteResponse *response) {
		captured = response;
		NFKServerRouteResponse *reply = response;
		response.disconnectHandler = ^{
			[reply sendStatus:500 headers:nil body:nil];
		};
		[response sendStatus:204 headers:nil body:nil];
	} error:NULL];

	NSHTTPURLResponse *response = nil;
	[self send:@"GET" to:@"/myapp/ping" body:nil headers:nil response:&response];
	XCTAssertEqual(response.statusCode, 204);
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:5];
	while (captured != nil && deadline.timeIntervalSinceNow > 0) {
		[NSRunLoop.currentRunLoop runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.01]];
	}
	XCTAssertNil(captured, @"the reply is released once it finishes");
}

- (void)testADisconnectHandlerSetAfterTheClientLeftRunsAtOnce
{
	__block NFKServerRouteResponse *held = nil;
	XCTestExpectation *arrived = [self expectationWithDescription:@"the request reaches the handler"];
	[self.server addRouteForPathPrefix:@"/myapp/" handler:^(NFKServerRouteRequest *request, NFKServerRouteResponse *response) {
		held = response;
		[arrived fulfill];
	} error:NULL];

	NSURLSession *session = [NSURLSession sessionWithConfiguration:NSURLSessionConfiguration.ephemeralSessionConfiguration];
	NSURLSessionDataTask *task = [session dataTaskWithURL:[self URLForPath:@"/myapp/hold"]];
	[task resume];
	[self waitForExpectations:@[ arrived ] timeout:5];
	[session invalidateAndCancel];
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:5];
	while (!held.isDisconnected && deadline.timeIntervalSinceNow > 0) {
		[NSRunLoop.currentRunLoop runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.01]];
	}
	XCTAssertTrue(held.isDisconnected);

	XCTestExpectation *fired = [self expectationWithDescription:@"the handler runs"];
	held.disconnectHandler = ^{
		[fired fulfill];
	};
	[self waitForExpectations:@[ fired ] timeout:5];
}

- (void)testAHandlerStreamsServerSentEvents
{
	[self.server addRouteForPathPrefix:@"/myapp/" handler:^(NFKServerRouteRequest *request, NFKServerRouteResponse *response) {
		[response beginStreamWithStatus:200 headers:@{ @"Content-Type": @"text/event-stream" }];
		for (NSUInteger index = 0; index < 3; index++) {
			[response sendEventData:[NSString stringWithFormat:@"{\"n\":%lu}", (unsigned long)index]];
		}
		[response endStream];
		XCTAssertTrue(response.isFinished);
	} error:NULL];

	NSHTTPURLResponse *response = nil;
	NSData *reply = [self send:@"GET" to:@"/myapp/events" body:nil headers:nil response:&response];
	XCTAssertEqual(response.statusCode, 200);
	XCTAssertEqualObjects([response valueForHTTPHeaderField:@"Content-Type"], @"text/event-stream");
	NSString *text = [[NSString alloc] initWithData:reply encoding:NSUTF8StringEncoding];
	XCTAssertEqualObjects(text, @"data: {\"n\":0}\n\ndata: {\"n\":1}\n\ndata: {\"n\":2}\n\n");
}

#pragma mark Prefixes

- (void)testTheServersOwnAndMalformedPrefixesAreRefused
{
	NSArray<NSString *> *refused = @[ @"/", @"", @"myapp", @"/v1", @"/v1/", @"/v1/myapp/", @"/V1/x", @"/mcp", @"/MCP/x/",
									   @"/a//b", @"/a/../b", @"/./a" ];
	for (NSString *prefix in refused) {
		NSError *error = nil;
		XCTAssertFalse([self.server addRouteForPathPrefix:prefix handler:[self echoHandlerNamed:@"x"] error:&error], @"%@", prefix);
		XCTAssertEqualObjects(error.domain, NFKInferenceServerErrorDomain, @"%@", prefix);
		XCTAssertEqual(error.code, NFKInferenceServerErrorInvalidConfiguration, @"%@", prefix);
	}
	XCTAssertEqualObjects(self.server.routePathPrefixes, @[]);
	XCTAssertTrue([self.server addRouteForPathPrefix:@"/v1beta/" handler:[self echoHandlerNamed:@"beta"] error:NULL],
				  @"a segment that only starts with v1 is the app's");

	NSHTTPURLResponse *response = nil;
	NSDictionary *reply = [self JSONFrom:[self send:@"GET" to:@"/v1beta/x" body:nil headers:nil response:&response]];
	XCTAssertEqualObjects(reply[@"handler"], @"beta");
	[self send:@"GET" to:@"/v1/nothing" body:nil headers:nil response:&response];
	XCTAssertEqual(response.statusCode, 404, @"an unknown path under /v1 stays the server's");
}

- (void)testTheLongestPrefixWinsAndARemovedRouteIsNotFound
{
	[self.server addRouteForPathPrefix:@"/myapp/" handler:[self echoHandlerNamed:@"app"] error:NULL];
	[self.server addRouteForPathPrefix:@"/myapp/admin/" handler:[self echoHandlerNamed:@"admin"] error:NULL];
	XCTAssertEqualObjects(self.server.routePathPrefixes, (@[ @"/myapp/", @"/myapp/admin/" ]));

	NSHTTPURLResponse *response = nil;
	XCTAssertEqualObjects([self JSONFrom:[self send:@"GET" to:@"/myapp/admin/users" body:nil headers:nil response:&response]][@"handler"], @"admin");
	XCTAssertEqualObjects([self JSONFrom:[self send:@"GET" to:@"/myapp/administer" body:nil headers:nil response:&response]][@"handler"], @"app",
						  @"a prefix matches whole segments");

	[self.server removeRouteForPathPrefix:@"/myapp/admin"];
	XCTAssertEqualObjects([self JSONFrom:[self send:@"GET" to:@"/myapp/admin/users" body:nil headers:nil response:&response]][@"handler"], @"app");
	[self.server removeRouteForPathPrefix:@"/myapp/"];
	[self send:@"GET" to:@"/myapp/admin/users" body:nil headers:nil response:&response];
	XCTAssertEqual(response.statusCode, 404);
	[self send:@"GET" to:@"/elsewhere" body:nil headers:nil response:&response];
	XCTAssertEqual(response.statusCode, 404);
}

#pragma mark Limits and access

- (void)testABodyOverTheServersLimitNeverReachesTheHandler
{
	[self.server stop];
	self.server.maximumRequestBodyBytes = 16;
	NSError *error = nil;
	XCTAssertTrue([self.server startWithError:&error], @"%@", error);
	__block BOOL called = NO;
	[self.server addRouteForPathPrefix:@"/myapp/" handler:^(NFKServerRouteRequest *request, NFKServerRouteResponse *response) {
		called = YES;
		[response sendStatus:200 headers:nil body:nil];
	} error:NULL];

	NSHTTPURLResponse *response = nil;
	[self send:@"POST" to:@"/myapp/upload" body:[NSMutableData dataWithLength:100] headers:nil response:&response];
	XCTAssertEqual(response.statusCode, 413);
	XCTAssertFalse(called);
	[self send:@"POST" to:@"/myapp/upload" body:[NSMutableData dataWithLength:16] headers:nil response:&response];
	XCTAssertEqual(response.statusCode, 200);
}

- (void)testAppRoutesTakeTheServersKey
{
	self.server.apiKey = @"secret";
	self.server.requiresAPIKeyOnLoopback = YES;
	__block BOOL called = NO;
	[self.server addRouteForPathPrefix:@"/myapp/" handler:^(NFKServerRouteRequest *request, NFKServerRouteResponse *response) {
		called = YES;
		[response sendStatus:204 headers:nil body:nil];
	} error:NULL];

	NSHTTPURLResponse *response = nil;
	[self send:@"GET" to:@"/myapp/x" body:nil headers:nil response:&response];
	XCTAssertEqual(response.statusCode, 401);
	XCTAssertFalse(called, @"a refused request never reaches the handler");
	[self send:@"GET" to:@"/myapp/x" body:nil headers:@{ @"Authorization": @"Bearer wrong" } response:&response];
	XCTAssertEqual(response.statusCode, 401);
	[self send:@"GET" to:@"/myapp/x" body:nil headers:@{ @"Authorization": @"Bearer secret" } response:&response];
	XCTAssertEqual(response.statusCode, 204);
	XCTAssertTrue(called);
}

@end
