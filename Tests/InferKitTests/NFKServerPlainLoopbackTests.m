//
//  NFKServerPlainLoopbackTests.m
//  InferKitTests
//
//  A TLS server's second listener, plain HTTP on loopback, over real connections on loopback ports: both
//  listeners serve the same models, counts, keys, and MCP sessions; the local URLs name the plain one;
//  the two start and stop together.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
#import <arpa/inet.h>
#import <sys/socket.h>
#import <unistd.h>

/*! A backend that answers every request with "ok". */
@interface NFKPlainLoopbackTestBackend : NSObject <NFKInferenceBackend>
@end

@implementation NFKPlainLoopbackTestBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"plain-loopback-test";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"ok" }];
}

@end

@interface NFKServerPlainLoopbackTests : XCTestCase
@property (nonatomic, strong) NFKServerIdentity *identity;
@property (nonatomic, strong) NFKInferenceServer *server;
@end

@implementation NFKServerPlainLoopbackTests

- (void)setUp
{
	[super setUp];
	NSError *error = nil;
	self.identity = [NFKServerIdentity createIdentityWithCommonName:nil subjectAlternativeNames:nil validityInterval:0 error:&error];
	XCTAssertNotNil(self.identity, @"%@", error);
	self.server = [[NFKInferenceServer alloc] init];
	self.server.port = 0;
	self.server.loopbackOnly = YES;
	self.server.TLSIdentity = self.identity.identity;
	self.server.servesPlainHTTPOnLoopback = YES;
	[self.server addBackend:[[NFKPlainLoopbackTestBackend alloc] init] forModelName:@"echo"];
}

- (void)tearDown
{
	[self.server stop];
	[super tearDown];
}

#pragma mark Helpers

- (void)start
{
	NSError *error = nil;
	XCTAssertTrue([self.server startWithError:&error], @"%@", error);
}

/*! The main listener's URL at path, as a client on this machine reaches it over TLS. */
- (NSURL *)TLSURLWithPath:(NSString *)path
{
	return [NSURL URLWithString:[NSString stringWithFormat:@"https://localhost:%u%@", (unsigned)self.server.listeningPort, path]];
}

- (NSURLSession *)pinnedSession
{
	return [NFKRemoteTransport sessionWithPinnedCertificateSHA256:self.identity.SHA256Fingerprint];
}

- (nullable NSString *)runOn:(NSURL *)baseURL pinned:(BOOL)pinned error:(NSError **)error
{
	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:baseURL];
	client.timeout = 10.0;
	client.pinnedCertificateSHA256 = pinned ? self.identity.SHA256Fingerprint : nil;
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hello" }];
	return [client runInferenceForRequest:request error:error].outputs[NFKOutputText];
}

/*! Sends method with body to url on session, presenting key, and returns the reply's head. */
- (nullable NSHTTPURLResponse *)send:(NSString *)method to:(NSURL *)url body:(nullable id)body
							 headers:(nullable NSDictionary<NSString *, NSString *> *)headers session:(NSURLSession *)session
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.HTTPMethod = method;
	if (body != nil) {
		request.HTTPBody = [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL];
		[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	}
	[request setValue:@"application/json, text/event-stream" forHTTPHeaderField:@"Accept"];
	[headers enumerateKeysAndObjectsUsingBlock:^(NSString *field, NSString *value, BOOL *stop) {
		[request setValue:value forHTTPHeaderField:field];
	}];
	NSHTTPURLResponse *response = nil;
	[NFKRemoteTransport sendRequest:request session:session response:&response error:NULL];
	return response;
}

/*! The status of a request written by hand to the plain listener, so its Host and Origin headers are the
	test's own; URL sessions replace Host. 0 when no reply arrives. */
- (NSInteger)statusOf:(NSString *)method path:(NSString *)path host:(NSString *)host
			  headers:(NSDictionary<NSString *, NSString *> *)headers body:(nullable NSString *)body
{
	int socketDescriptor = socket(AF_INET, SOCK_STREAM, 0);
	XCTAssertGreaterThanOrEqual(socketDescriptor, 0);
	struct timeval timeout = { 5, 0 };
	setsockopt(socketDescriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
	struct sockaddr_in address = { 0 };
	address.sin_len = sizeof(address);
	address.sin_family = AF_INET;
	address.sin_port = htons(self.server.plainLoopbackListeningPort);
	address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	if (connect(socketDescriptor, (struct sockaddr *)&address, sizeof(address)) != 0) {
		close(socketDescriptor);
		return 0;
	}
	NSMutableString *text = [NSMutableString stringWithFormat:@"%@ %@ HTTP/1.1\r\nHost: %@\r\nConnection: close\r\n", method, path, host];
	[headers enumerateKeysAndObjectsUsingBlock:^(NSString *field, NSString *value, BOOL *stop) {
		[text appendFormat:@"%@: %@\r\n", field, value];
	}];
	NSData *bodyData = [body dataUsingEncoding:NSUTF8StringEncoding] ?: [NSData data];
	[text appendFormat:@"Content-Length: %lu\r\n\r\n", (unsigned long)bodyData.length];
	NSMutableData *message = [[text dataUsingEncoding:NSUTF8StringEncoding] mutableCopy];
	[message appendData:bodyData];
	write(socketDescriptor, message.bytes, message.length);
	char reply[64] = { 0 };
	ssize_t length = read(socketDescriptor, reply, sizeof(reply) - 1);
	close(socketDescriptor);
	if (length < 12) {
		return 0;
	}
	return [[[NSString alloc] initWithBytes:reply + 9 length:3 encoding:NSASCIIStringEncoding] integerValue];
}

#pragma mark Listeners

- (void)testBothListenersServeTheSameModels
{
	[self start];
	XCTAssertGreaterThan(self.server.plainLoopbackListeningPort, 0);
	XCTAssertNotEqual(self.server.plainLoopbackListeningPort, self.server.listeningPort);
	XCTAssertEqualObjects(self.server.localBaseURL.scheme, @"http");
	XCTAssertEqualObjects(self.server.localBaseURL.port, @(self.server.plainLoopbackListeningPort));
	XCTAssertEqualObjects(self.server.localMCPURL.absoluteString,
						  ([NSString stringWithFormat:@"http://localhost:%u/mcp", (unsigned)self.server.plainLoopbackListeningPort]));

	NSError *error = nil;
	XCTAssertEqualObjects([self runOn:self.server.localBaseURL pinned:NO error:&error], @"ok", @"%@", error);
	XCTAssertEqualObjects([self runOn:[self TLSURLWithPath:@"/v1"] pinned:YES error:&error], @"ok", @"%@", error);
	XCTAssertEqual([self.server.currentStatus modelNamed:@"echo"].completed, 2, @"one model serves both listeners");
	XCTAssertNil([self runOn:[self TLSURLWithPath:@"/v1"] pinned:NO error:&error], @"the main listener still needs the pin");
}

- (void)testWithoutThePlainListenerTheLocalURLsNameTheTLSOne
{
	self.server.servesPlainHTTPOnLoopback = NO;
	[self start];
	XCTAssertEqual(self.server.plainLoopbackListeningPort, 0);
	XCTAssertEqualObjects(self.server.localBaseURL.scheme, @"https");
	XCTAssertEqualObjects(self.server.localBaseURL.port, @(self.server.listeningPort));
	XCTAssertNil(self.server.networkBaseURL, @"a loopback-only server has no network URL");
	XCTAssertNil(self.server.networkMCPURL);
}

- (void)testTheNetworkURLsNameTheTLSListener
{
	self.server.loopbackOnly = NO;
	self.server.requiresAPIKey = NO;
	self.server.advertisesService = NO;
	[self start];
	NSURL *network = self.server.networkBaseURL;
	XCTAssertEqualObjects(network.scheme, @"https");
	XCTAssertEqualObjects(network.port, @(self.server.listeningPort));
	XCTAssertEqualObjects(network.path, @"/v1");
	XCTAssertGreaterThan(network.host.length, 0);
	XCTAssertEqualObjects(self.server.networkMCPURL.path, @"/mcp");
	XCTAssertEqualObjects(self.server.localBaseURL.port, @(self.server.plainLoopbackListeningPort));
}

- (void)testStopStopsBothAndATakenPlainPortStartsNeither
{
	[self start];
	uint16_t plainPort = self.server.plainLoopbackListeningPort;
	NFKInferenceServer *second = [[NFKInferenceServer alloc] init];
	second.port = 0;
	second.loopbackOnly = YES;
	second.servesPlainHTTPOnLoopback = YES;
	second.plainLoopbackPort = plainPort;
	NSError *error = nil;
	XCTAssertFalse([second startWithError:&error]);
	XCTAssertEqual(error.code, NFKInferenceServerErrorListenFailed);
	XCTAssertFalse(second.isRunning);
	XCTAssertEqual(second.listeningPort, 0, @"the main listener does not stay up alone");

	[self.server stop];
	XCTAssertEqual(self.server.plainLoopbackListeningPort, 0);
	XCTAssertNil(self.server.localBaseURL);
	XCTAssertTrue([second startWithError:&error], @"the stopped server released the plain port: %@", error);
	XCTAssertEqual(second.plainLoopbackListeningPort, plainPort);
	[second stop];
}

#pragma mark Host checks

- (void)testAForeignHostNeedsAKeyOnThePlainListener
{
	[self.server addAPIKey:@"key-one" named:@"one" scopes:NFKServerKeyScopeInference];
	[self start];
	NSString *port = [NSString stringWithFormat:@"%u", (unsigned)self.server.plainLoopbackListeningPort];
	XCTAssertEqual([self statusOf:@"GET" path:@"/v1/models" host:[@"attacker.example:" stringByAppendingString:port] headers:@{} body:nil], 421,
				   @"a rebound browser page gets no key-free access");
	XCTAssertEqual([self statusOf:@"GET" path:@"/v1/inferkit/status" host:@"attacker.example" headers:@{} body:nil], 421);
	XCTAssertEqual([self statusOf:@"GET" path:@"/v1/models" host:[@"localhost:" stringByAppendingString:port] headers:@{} body:nil], 200);
	XCTAssertEqual([self statusOf:@"GET" path:@"/v1/models" host:@"127.0.0.1" headers:@{} body:nil], 200);
	XCTAssertEqual([self statusOf:@"GET" path:@"/v1/models" host:[@"[::1]:" stringByAppendingString:port] headers:@{} body:nil], 200);
	NSDictionary *key = @{ @"Authorization": @"Bearer key-one" };
	XCTAssertEqual([self statusOf:@"GET" path:@"/v1/models" host:@"attacker.example" headers:key body:nil], 200,
				   @"a key that opens the route serves any Host, as a reverse proxy sends");
	XCTAssertEqual([self statusOf:@"GET" path:@"/v1/inferkit/status" host:@"attacker.example" headers:key body:nil], 421,
				   @"the key must open the route");
}

- (void)testTheMCPOriginCheckStillRefusesAnotherSite
{
	[self start];
	NSString *host = [NSString stringWithFormat:@"localhost:%u", (unsigned)self.server.plainLoopbackListeningPort];
	NSString *ping = @"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}";
	NSDictionary *json = @{ @"Content-Type": @"application/json", @"Accept": @"application/json, text/event-stream" };
	NSMutableDictionary *foreign = [json mutableCopy];
	foreign[@"Origin"] = @"http://attacker.example";
	NSMutableDictionary *local = [json mutableCopy];
	local[@"Origin"] = [@"http://" stringByAppendingString:host];
	XCTAssertEqual([self statusOf:@"POST" path:@"/mcp" host:host headers:foreign body:ping], 403);
	XCTAssertEqual([self statusOf:@"POST" path:@"/mcp" host:host headers:local body:ping], 200);
	XCTAssertEqual([self statusOf:@"POST" path:@"/mcp" host:host headers:json body:ping], 200, @"an agent CLI sends no Origin");
}

#pragma mark Shared state

- (void)testKeysApplyOnBothListeners
{
	self.server.requiresAPIKeyOnLoopback = YES;
	[self.server addAPIKey:@"key-one" named:@"one" scopes:NFKServerKeyScopeAll];
	[self start];
	NSURL *plainModels = [self.server.localBaseURL URLByAppendingPathComponent:@"models"];
	NSURL *TLSModels = [self TLSURLWithPath:@"/v1/models"];
	NSDictionary *key = @{ @"Authorization": @"Bearer key-one" };
	XCTAssertEqual([self send:@"GET" to:plainModels body:nil headers:nil session:NSURLSession.sharedSession].statusCode, 401);
	XCTAssertEqual([self send:@"GET" to:plainModels body:nil headers:key session:NSURLSession.sharedSession].statusCode, 200);
	XCTAssertEqual([self send:@"GET" to:TLSModels body:nil headers:key session:self.pinnedSession].statusCode, 200);

	[self.server removeAPIKeyNamed:@"one"];
	XCTAssertEqual([self send:@"GET" to:plainModels body:nil headers:key session:NSURLSession.sharedSession].statusCode, 401);
	XCTAssertEqual([self send:@"GET" to:TLSModels body:nil headers:key session:self.pinnedSession].statusCode, 401);
}

- (void)testAnMCPSessionIssuedOnOneListenerStartsOnTheOther
{
	[self start];
	NSDictionary *initialize = @{ @"jsonrpc": @"2.0", @"id": @1, @"method": @"initialize",
								  @"params": @{ @"protocolVersion": @"2025-06-18",
												@"clientInfo": @{ @"name": @"agent", @"version": @"1.0" } } };
	NSHTTPURLResponse *issued = [self send:@"POST" to:self.server.localMCPURL body:initialize headers:nil session:NSURLSession.sharedSession];
	XCTAssertEqual(issued.statusCode, 200);
	NSString *identifier = [issued valueForHTTPHeaderField:@"Mcp-Session-Id"];
	XCTAssertGreaterThan(identifier.length, 0);

	NSDictionary *ping = @{ @"jsonrpc": @"2.0", @"id": @2, @"method": @"ping" };
	NSHTTPURLResponse *confirmed = [self send:@"POST" to:[self TLSURLWithPath:@"/mcp"] body:ping
									  headers:@{ @"Mcp-Session-Id": identifier } session:self.pinnedSession];
	XCTAssertEqual(confirmed.statusCode, 200);
	XCTAssertEqual(self.server.MCPSessions.count, 1);
	XCTAssertEqualObjects(self.server.MCPSessions.firstObject.clientName, @"agent");
}

@end
