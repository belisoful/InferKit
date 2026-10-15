//
//  NFKServedMCPTests.m
//  InferKitTests
//
//  The server's MCP endpoint over real HTTP on a loopback port: the handshake, the registered tools
//  and nothing else, a call's content, the protocol's errors and acknowledgements, and the access
//  rules the other routes keep.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

@interface NFKServedMCPTests : XCTestCase
@property (nonatomic, strong) NFKInferenceServer *server;
@end

@implementation NFKServedMCPTests

- (void)setUp
{
	[super setUp];
	self.server = [[NFKInferenceServer alloc] init];
	self.server.port = 0;
	self.server.loopbackOnly = YES;
	[self.server addBackend:[[NFKPassthroughBackend alloc] init] forModelName:@"echo"];
	NSError *error = nil;
	XCTAssertTrue([self.server startWithError:&error], @"%@", error);
}

- (void)tearDown
{
	[self.server stop];
	[super tearDown];
}

- (NFKInferenceTool *)weatherTool
{
	return [NFKInferenceTool toolWithName:@"get_weather" description:@"Current weather in a city."
							   parameters:@{ @"type": @"object", @"properties": @{ @"city": @{ @"type": @"string" } }, @"required": @[ @"city" ] }
								  handler:^NFKToolResult *(NSDictionary *arguments, NSError **error) {
		return [NFKToolResult resultWithText:[NSString stringWithFormat:@"18 C in %@", arguments[@"city"]]];
	}];
}

/*! Posts body to the endpoint and returns the parsed reply, the status, and the reply's headers. */
- (nullable id)post:(id)body headers:(nullable NSDictionary<NSString *, NSString *> *)headers
			 status:(NSInteger *)outStatus response:(NSHTTPURLResponse **)outResponse
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:self.server.localMCPURL];
	request.HTTPMethod = @"POST";
	request.HTTPBody = [body isKindOfClass:NSData.class] ? body : [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL];
	[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	[request setValue:@"application/json, text/event-stream" forHTTPHeaderField:@"Accept"];
	[headers enumerateKeysAndObjectsUsingBlock:^(NSString *field, NSString *value, BOOL *stop) {
		[request setValue:value forHTTPHeaderField:field];
	}];
	NSHTTPURLResponse *response = nil;
	NSData *data = [NFKRemoteTransport sendRequest:request session:NSURLSession.sharedSession response:&response error:NULL];
	if (outStatus != NULL) {
		*outStatus = response.statusCode;
	}
	if (outResponse != NULL) {
		*outResponse = response;
	}
	return data.length > 0 ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
}

- (NSDictionary *)call:(NSString *)method params:(NSDictionary *)params
{
	NSInteger status = 0;
	NSDictionary *reply = [self post:@{ @"jsonrpc": @"2.0", @"id": @7, @"method": method, @"params": params } headers:nil status:&status response:NULL];
	XCTAssertEqual(status, 200, @"%@", method);
	XCTAssertEqualObjects(reply[@"id"], @7);
	return reply;
}

#pragma mark The handshake

- (void)testInitializeAgreesOnAVersionAndNamesTheServer
{
	self.server.MCPInstructions = @"Prefer get_weather for anything about weather.";
	NSDictionary *agreed = [self call:@"initialize" params:@{ @"protocolVersion": @"2025-06-18", @"capabilities": @{},
															   @"clientInfo": @{ @"name": @"test", @"version": @"1" } }][@"result"];
	XCTAssertEqualObjects(agreed[@"protocolVersion"], @"2025-06-18", @"a version the server speaks is the one it answers with");
	XCTAssertEqualObjects(agreed[@"serverInfo"][@"name"], @"InferKit");
	XCTAssertEqualObjects(agreed[@"serverInfo"][@"version"], NFKInferKit.releaseVersion);
	XCTAssertEqualObjects(agreed[@"capabilities"][@"tools"][@"listChanged"], @NO);
	XCTAssertEqualObjects(agreed[@"instructions"], @"Prefer get_weather for anything about weather.");

	NSDictionary *newest = [self call:@"initialize" params:@{ @"protocolVersion": @"1999-01-01" }][@"result"];
	XCTAssertEqualObjects(newest[@"protocolVersion"], @"2025-11-25", @"otherwise the newest it speaks");
	XCTAssertEqualObjects([self call:@"ping" params:@{}][@"result"], @{});
	XCTAssertEqualObjects(self.server.localMCPURL.path, @"/mcp");
}

#pragma mark The tools

- (void)testOnlyTheRegisteredToolsAreListed
{
	XCTAssertEqualObjects([self call:@"tools/list" params:@{}][@"result"][@"tools"], @[], @"a hosted model is not a tool until one is added");
	NFKInferenceTool *weather = [self weatherTool];
	[self.server addTool:weather];
	NSArray *listed = [self call:@"tools/list" params:@{}][@"result"][@"tools"];
	XCTAssertEqual(listed.count, 1u);
	XCTAssertEqualObjects(listed[0][@"name"], @"get_weather");
	XCTAssertEqualObjects(listed[0][@"description"], @"Current weather in a city.");
	XCTAssertEqualObjects(listed[0][@"inputSchema"], weather.parameters);

	NFKInferenceTool *replacement = [NFKInferenceTool toolWithName:@"get_weather" description:@"Replaced." parameters:@{ @"type": @"object" }
														   handler:^NFKToolResult *(NSDictionary *arguments, NSError **error) { return [NFKToolResult resultWithText:@"x"]; }];
	[self.server addTool:replacement];
	XCTAssertEqual(self.server.tools.count, 1u, @"a tool of the same name replaces the first");
	XCTAssertEqualObjects([self call:@"tools/list" params:@{}][@"result"][@"tools"][0][@"description"], @"Replaced.");
	[self.server removeToolNamed:@"get_weather"];
	XCTAssertEqualObjects(self.server.tools, @[]);
}

- (void)testACallReturnsItsTextImagesAndAudioAsContent
{
	NSURL *clip = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:[NSUUID.UUID.UUIDString stringByAppendingString:@".wav"]];
	NSData *wav = [@"RIFF....WAVE" dataUsingEncoding:NSUTF8StringEncoding];
	[wav writeToURL:clip atomically:YES];
	[self addTeardownBlock:^{
		[NSFileManager.defaultManager removeItemAtURL:clip error:NULL];
	}];
	NSData *png = [NSData dataWithBytes:"\x89PNG\r\n" length:6];
	[self.server addTool:[NFKInferenceTool toolWithName:@"speak" description:@"Speaks." parameters:@{ @"type": @"object" }
												handler:^NFKToolResult *(NSDictionary *arguments, NSError **error) {
		return [NFKToolResult resultWithText:@"Generated 0.5 seconds of audio."
									 outputs:@{ NFKOutputAudio: [NFKAudioAsset audioAssetWithFileURL:clip],
												NFKOutputStructured: @{ @"seconds": @0.5 } }
								   imagePNGs:@[ png ]];
	}]];
	NSDictionary *result = [self call:@"tools/call" params:@{ @"name": @"speak", @"arguments": @{} }][@"result"];
	NSArray *content = result[@"content"];
	XCTAssertEqualObjects(content[0], (@{ @"type": @"text", @"text": @"Generated 0.5 seconds of audio." }));
	XCTAssertEqualObjects(content[1][@"type"], @"image");
	XCTAssertEqualObjects(content[1][@"mimeType"], @"image/png");
	XCTAssertEqualObjects([[NSData alloc] initWithBase64EncodedString:content[1][@"data"] options:0], png);
	XCTAssertEqualObjects(content[2][@"type"], @"audio");
	XCTAssertEqualObjects(content[2][@"mimeType"], @"audio/wav");
	XCTAssertEqualObjects([[NSData alloc] initWithBase64EncodedString:content[2][@"data"] options:0], wav);
	XCTAssertEqualObjects(result[@"isError"], @NO);
	XCTAssertEqualObjects(result[@"structuredContent"], @{ @"seconds": @0.5 });
}

- (void)testAFailingCallIsAnErrorResultAndAnUnknownToolAProtocolError
{
	[self.server addTool:[self weatherTool]];
	NSDictionary *failed = [self call:@"tools/call" params:@{ @"name": @"get_weather", @"arguments": @{} }][@"result"];
	XCTAssertEqualObjects(failed[@"isError"], @YES, @"a call the tool refuses reaches the model as its result");
	XCTAssertTrue([failed[@"content"][0][@"text"] containsString:@"\"city\" is missing"]);

	NSDictionary *unknown = [self call:@"tools/call" params:@{ @"name": @"open_door", @"arguments": @{} }];
	XCTAssertEqualObjects(unknown[@"error"][@"code"], @(-32602));
	XCTAssertEqualObjects([self call:@"resources/list" params:@{}][@"error"][@"code"], @(-32601));
}

#pragma mark The transport

- (void)testANotificationIsAcknowledgedAndABatchIsAnsweredInOrder
{
	NSInteger status = 0;
	id reply = [self post:@{ @"jsonrpc": @"2.0", @"method": @"notifications/initialized" } headers:nil status:&status response:NULL];
	XCTAssertEqual(status, 202);
	XCTAssertNil(reply);

	NSArray *batch = @[ @{ @"jsonrpc": @"2.0", @"id": @1, @"method": @"ping" },
						@{ @"jsonrpc": @"2.0", @"method": @"notifications/initialized" },
						@{ @"jsonrpc": @"2.0", @"id": @2, @"method": @"tools/list" } ];
	NSArray *replies = [self post:batch headers:nil status:&status response:NULL];
	XCTAssertEqual(status, 200);
	XCTAssertEqual(replies.count, 2u);
	XCTAssertEqualObjects(replies[0][@"id"], @1);
	XCTAssertEqualObjects(replies[1][@"id"], @2);

	NSDictionary *notJSON = [self post:[@"{" dataUsingEncoding:NSUTF8StringEncoding] headers:nil status:&status response:NULL];
	XCTAssertEqual(status, 400);
	XCTAssertEqualObjects(notJSON[@"error"][@"code"], @(-32700));
}

- (void)testAStreamWithoutASessionIsRefusedAndSoIsAnUnknownVersion
{
	NSMutableURLRequest *get = [NSMutableURLRequest requestWithURL:self.server.localMCPURL];
	NSHTTPURLResponse *response = nil;
	[NFKRemoteTransport sendRequest:get session:NSURLSession.sharedSession response:&response error:NULL];
	XCTAssertEqual(response.statusCode, 405, @"a stream needs the session initialize issued");
	XCTAssertEqualObjects([response valueForHTTPHeaderField:@"Allow"], @"GET, POST, DELETE");

	NSInteger status = 0;
	[self post:@{ @"jsonrpc": @"2.0", @"id": @1, @"method": @"ping" } headers:@{ @"MCP-Protocol-Version": @"2099-01-01" } status:&status response:NULL];
	XCTAssertEqual(status, 400);
	[self post:@{ @"jsonrpc": @"2.0", @"id": @1, @"method": @"ping" } headers:@{ @"MCP-Protocol-Version": @"2025-06-18" } status:&status response:NULL];
	XCTAssertEqual(status, 200);
}

#pragma mark Access

- (void)testABrowserPageOnAnotherHostIsRefused
{
	NSInteger status = 0;
	NSDictionary *ping = @{ @"jsonrpc": @"2.0", @"id": @1, @"method": @"ping" };
	[self post:ping headers:@{ @"Origin": @"https://attacker.example" } status:&status response:NULL];
	XCTAssertEqual(status, 403, @"DNS rebinding reaches a loopback server from another site's page");
	[self post:ping headers:@{ @"Origin": @"http://localhost:3000" } status:&status response:NULL];
	XCTAssertEqual(status, 200);
	[self post:ping headers:nil status:&status response:NULL];
	XCTAssertEqual(status, 200, @"a client that is not a browser sends no Origin");
}

- (void)testTheEndpointTakesTheServersKey
{
	self.server.apiKey = @"secret";
	self.server.requiresAPIKeyOnLoopback = YES;
	NSInteger status = 0;
	NSDictionary *ping = @{ @"jsonrpc": @"2.0", @"id": @1, @"method": @"ping" };
	[self post:ping headers:nil status:&status response:NULL];
	XCTAssertEqual(status, 401);
	[self post:ping headers:@{ @"Authorization": @"Bearer secret" } status:&status response:NULL];
	XCTAssertEqual(status, 200);
}

@end
