//
//  NFKServerKeyTests.m
//  InferKitTests
//
//  Named, scoped, revocable API keys over real HTTP on a loopback port: each scope opens only its
//  routes, a revoked key is refused at once, loopback exemption is per scope, and apiKey is the key
//  named "default" with every scope.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

@interface NFKServerKeyTests : XCTestCase
@property (nonatomic, strong) NFKInferenceServer *server;
@end

@implementation NFKServerKeyTests

- (void)setUp
{
	[super setUp];
	self.server = [[NFKInferenceServer alloc] init];
	self.server.port = 0;
	self.server.loopbackOnly = YES;
	[self.server addBackend:[[NFKPassthroughBackend alloc] init] forModelName:@"echo"];
	[self.server addRouteForPathPrefix:@"/myapp/" handler:^(NFKServerRouteRequest *request, NFKServerRouteResponse *response) {
		[response sendStatus:200 JSONObject:@{ @"key": request.keyName ?: NSNull.null } headers:nil];
	} error:NULL];
	NSError *error = nil;
	XCTAssertTrue([self.server startWithError:&error], @"%@", error);
}

- (void)tearDown
{
	[self.server stop];
	[super tearDown];
}

#pragma mark Helpers

/*! The status of one request to a route of each scope, presenting key (nil for none), keyed by scope. */
- (NSDictionary<NSNumber *, NSNumber *> *)statusesWithKey:(nullable NSString *)key
{
	NSMutableDictionary<NSNumber *, NSNumber *> *statuses = [NSMutableDictionary dictionary];
	statuses[@(NFKServerKeyScopeInference)] = @([self statusOf:@"GET" path:@"/v1/models" body:nil key:key]);
	statuses[@(NFKServerKeyScopeStatus)] = @([self statusOf:@"GET" path:@"/v1/inferkit/status" body:nil key:key]);
	NSData *ping = [NSJSONSerialization dataWithJSONObject:@{ @"jsonrpc": @"2.0", @"id": @1, @"method": @"ping" } options:0 error:NULL];
	statuses[@(NFKServerKeyScopeMCP)] = @([self statusOf:@"POST" path:@"/mcp" body:ping key:key]);
	statuses[@(NFKServerKeyScopeAppRoutes)] = @([self statusOf:@"GET" path:@"/myapp/x" body:nil key:key]);
	return statuses;
}

/*! The statuses statusesWithKey: expects when exactly the scopes in open answer and the others refuse
	with refusal. */
- (NSDictionary<NSNumber *, NSNumber *> *)statusesOpening:(NFKServerKeyScope)open refusing:(NSInteger)refusal
{
	NSMutableDictionary<NSNumber *, NSNumber *> *statuses = [NSMutableDictionary dictionary];
	for (NSNumber *scope in @[ @(NFKServerKeyScopeInference), @(NFKServerKeyScopeStatus), @(NFKServerKeyScopeMCP), @(NFKServerKeyScopeAppRoutes) ]) {
		statuses[scope] = (open & scope.unsignedIntegerValue) != 0 ? @200 : @(refusal);
	}
	return statuses;
}

- (NSInteger)statusOf:(NSString *)method path:(NSString *)path body:(nullable NSData *)body key:(nullable NSString *)key
{
	return [self responseTo:method path:path body:body key:key data:NULL].statusCode;
}

- (nullable NSHTTPURLResponse *)responseTo:(NSString *)method path:(NSString *)path body:(nullable NSData *)body
									   key:(nullable NSString *)key data:(NSData * _Nullable * _Nullable)outData
{
	NSURL *url = [NSURL URLWithString:[NSString stringWithFormat:@"http://localhost:%u%@", (unsigned)self.server.listeningPort, path]];
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.HTTPMethod = method;
	request.HTTPBody = body;
	if (body != nil) {
		[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	}
	if (key != nil) {
		[request setValue:[@"Bearer " stringByAppendingString:key] forHTTPHeaderField:@"Authorization"];
	}
	NSHTTPURLResponse *response = nil;
	NSData *data = [NFKRemoteTransport sendRequest:request session:NSURLSession.sharedSession response:&response error:NULL];
	if (outData != NULL) {
		*outData = data;
	}
	return response;
}

#pragma mark Scopes

- (void)testEachScopeOpensOnlyItsRoutes
{
	self.server.loopbackExemptScopes = 0;
	[self.server addAPIKey:@"key-inference" named:@"inference" scopes:NFKServerKeyScopeInference];
	[self.server addAPIKey:@"key-status" named:@"status" scopes:NFKServerKeyScopeStatus];
	[self.server addAPIKey:@"key-mcp" named:@"mcp" scopes:NFKServerKeyScopeMCP];
	[self.server addAPIKey:@"key-app" named:@"app" scopes:NFKServerKeyScopeAppRoutes];
	[self.server addAPIKey:@"key-balancer" named:@"balancer" scopes:NFKServerKeyScopeInference | NFKServerKeyScopeStatus];
	XCTAssertEqualObjects(self.server.apiKeyNames, (@[ @"app", @"balancer", @"inference", @"mcp", @"status" ]));
	XCTAssertEqual([self.server scopesForAPIKeyNamed:@"balancer"], NFKServerKeyScopeInference | NFKServerKeyScopeStatus);
	XCTAssertEqual([self.server scopesForAPIKeyNamed:@"absent"], 0);

	XCTAssertEqualObjects([self statusesWithKey:@"key-inference"], [self statusesOpening:NFKServerKeyScopeInference refusing:403]);
	XCTAssertEqualObjects([self statusesWithKey:@"key-status"], [self statusesOpening:NFKServerKeyScopeStatus refusing:403]);
	XCTAssertEqualObjects([self statusesWithKey:@"key-mcp"], [self statusesOpening:NFKServerKeyScopeMCP refusing:403]);
	XCTAssertEqualObjects([self statusesWithKey:@"key-app"], [self statusesOpening:NFKServerKeyScopeAppRoutes refusing:403]);
	XCTAssertEqualObjects([self statusesWithKey:@"key-balancer"], [self statusesOpening:NFKServerKeyScopeInference | NFKServerKeyScopeStatus refusing:403]);
	XCTAssertEqualObjects([self statusesWithKey:nil], [self statusesOpening:0 refusing:401]);
	XCTAssertEqualObjects([self statusesWithKey:@"key-app-and-more"], [self statusesOpening:0 refusing:401], @"a key must match whole");

	NSData *data = nil;
	[self responseTo:@"GET" path:@"/myapp/x" body:nil key:@"key-app" data:&data];
	NSDictionary *reply = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	XCTAssertEqualObjects(reply[@"key"], @"app", @"an app route learns the key's name");
	NSHTTPURLResponse *unknown = [self responseTo:@"GET" path:@"/v1/models" body:nil key:@"key-absent" data:NULL];
	XCTAssertEqualObjects([unknown valueForHTTPHeaderField:@"WWW-Authenticate"], @"Bearer");
	NSHTTPURLResponse *refused = [self responseTo:@"GET" path:@"/v1/models" body:nil key:@"key-app" data:&data];
	XCTAssertEqualObjects([refused valueForHTTPHeaderField:@"WWW-Authenticate"], @"Bearer error=\"insufficient_scope\"");
	NSString *text = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
	XCTAssertFalse([text containsString:@"key-app"], @"a refusal does not echo the key");
	XCTAssertFalse([text containsString:@"\"app\""], @"a refusal does not name the key");
}

- (void)testARevokedKeyIsRefusedOnTheNextRequest
{
	self.server.loopbackExemptScopes = 0;
	[self.server addAPIKey:@"key-one" named:@"one" scopes:NFKServerKeyScopeAll];
	[self.server addAPIKey:@"key-two" named:@"two" scopes:NFKServerKeyScopeAll];
	XCTAssertEqual([self statusOf:@"GET" path:@"/v1/models" body:nil key:@"key-one"], 200);

	[self.server removeAPIKeyNamed:@"one"];
	XCTAssertEqualObjects(self.server.apiKeyNames, @[ @"two" ]);
	XCTAssertEqual([self statusOf:@"GET" path:@"/v1/models" body:nil key:@"key-one"], 401);
	XCTAssertEqual([self statusOf:@"GET" path:@"/v1/models" body:nil key:@"key-two"], 200, @"the other key stays");

	[self.server addAPIKey:@"key-replaced" named:@"two" scopes:NFKServerKeyScopeStatus];
	XCTAssertEqual([self statusOf:@"GET" path:@"/v1/models" body:nil key:@"key-two"], 401, @"adding a name again replaces its key");
	XCTAssertEqual([self statusOf:@"GET" path:@"/v1/inferkit/status" body:nil key:@"key-replaced"], 200);
}

#pragma mark Loopback

- (void)testLoopbackExemptionIsPerScope
{
	XCTAssertEqual(self.server.loopbackExemptScopes, NFKServerKeyScopeAll);
	XCTAssertFalse(self.server.requiresAPIKeyOnLoopback);
	XCTAssertEqualObjects([self statusesWithKey:nil], [self statusesOpening:NFKServerKeyScopeAll refusing:401]);

	[self.server addAPIKey:@"key-inference" named:@"inference" scopes:NFKServerKeyScopeInference];
	self.server.loopbackExemptScopes = NFKServerKeyScopeStatus | NFKServerKeyScopeAppRoutes;
	XCTAssertTrue(self.server.requiresAPIKeyOnLoopback, @"some scope needs a key on loopback");
	XCTAssertEqualObjects([self statusesWithKey:nil], [self statusesOpening:NFKServerKeyScopeStatus | NFKServerKeyScopeAppRoutes refusing:401]);
	XCTAssertEqualObjects([self statusesWithKey:@"key-inference"],
						  [self statusesOpening:NFKServerKeyScopeStatus | NFKServerKeyScopeAppRoutes | NFKServerKeyScopeInference refusing:403]);

	self.server.requiresAPIKeyOnLoopback = NO;
	XCTAssertEqual(self.server.loopbackExemptScopes, NFKServerKeyScopeAll);
	self.server.requiresAPIKeyOnLoopback = YES;
	XCTAssertEqual(self.server.loopbackExemptScopes, 0);
	XCTAssertEqualObjects([self statusesWithKey:nil], [self statusesOpening:0 refusing:401]);
}

- (void)testStartRequiresAKeyWhenAScopeNeedsOne
{
	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.port = 0;
	server.loopbackOnly = YES;
	server.loopbackExemptScopes = NFKServerKeyScopeAll & ~NFKServerKeyScopeMCP;
	NSError *error = nil;
	XCTAssertFalse([server startWithError:&error]);
	XCTAssertEqual(error.code, NFKInferenceServerErrorAPIKeyRequired);

	[server addAPIKey:@"key-mcp" named:@"agents" scopes:NFKServerKeyScopeMCP];
	XCTAssertTrue([server startWithError:&error], @"%@", error);
	[server stop];
}

#pragma mark The single key

- (void)testApiKeyIsTheDefaultKeyWithEveryScope
{
	self.server.requiresAPIKeyOnLoopback = YES;
	self.server.apiKey = @"legacy";
	XCTAssertEqualObjects(self.server.apiKeyNames, @[ NFKInferenceServerDefaultAPIKeyName ]);
	XCTAssertEqualObjects(NFKInferenceServerDefaultAPIKeyName, @"default");
	XCTAssertEqual([self.server scopesForAPIKeyNamed:@"default"], NFKServerKeyScopeAll);
	XCTAssertEqualObjects([self statusesWithKey:@"legacy"], [self statusesOpening:NFKServerKeyScopeAll refusing:403]);

	[self.server addAPIKey:@"narrowed" named:NFKInferenceServerDefaultAPIKeyName scopes:NFKServerKeyScopeMCP];
	XCTAssertEqualObjects(self.server.apiKey, @"narrowed", @"apiKey reads the key named default");
	XCTAssertEqualObjects([self statusesWithKey:@"narrowed"], [self statusesOpening:NFKServerKeyScopeMCP refusing:403]);

	self.server.apiKey = nil;
	XCTAssertEqualObjects(self.server.apiKeyNames, @[]);
	XCTAssertEqualObjects([self statusesWithKey:@"narrowed"], [self statusesOpening:0 refusing:401]);
}

@end
