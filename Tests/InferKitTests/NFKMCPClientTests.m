//
//  NFKMCPClientTests.m
//  InferKitTests
//
//  The MCP client against the core's own server on a loopback port: a server's tools as
//  NFKInferenceTool, their media read back into the contract's types, a tool loop over a remote
//  server's tools, and a server's errors. A stubbed transport stands in for a server that answers in
//  an event stream and assigns a session.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

/*! A model that calls the first tool it is offered, then answers with what the tool returned. */
@interface NFKMCPCallingModel : NSObject <NFKInferenceBackend>
@end

@implementation NFKMCPCallingModel
- (BOOL)isReady { return YES; }
- (NSString *)backendIdentifier { return @"mcp-calling"; }
- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	NSDictionary *last = request.messages.lastObject;
	if ([last[@"role"] isEqual:@"tool"]) {
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: [@"Remote says: " stringByAppendingString:last[@"content"]] }];
	}
	NSString *name = [request.parameters[NFKParameterTools] firstObject][@"name"];
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputToolCalls: @[ @{ @"id": @"c1", @"name": name, @"arguments": @{ NFKInputPrompt: @"Weather in Paris?" } } ] }];
}
@end

/*! Answers every MCP request in an event stream, assigns a session on initialize, and records the
	headers each request carried. */
@interface NFKMCPEventStreamProtocol : NSURLProtocol
@property (class, nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@end

@implementation NFKMCPEventStreamProtocol

static NSMutableArray<NSURLRequest *> *NFKMCPRecordedRequests;

+ (NSMutableArray<NSURLRequest *> *)requests { return NFKMCPRecordedRequests; }
+ (void)setRequests:(NSMutableArray<NSURLRequest *> *)requests { NFKMCPRecordedRequests = requests; }
+ (BOOL)canInitWithRequest:(NSURLRequest *)request { return [request.URL.host isEqualToString:@"mcp.stub.invalid"]; }
+ (NSURLRequest *)canonicalRequestForRequest:(NSURLRequest *)request { return request; }

- (void)startLoading
{
	NSData *body = self.request.HTTPBody;
	if (body == nil && self.request.HTTPBodyStream != nil) {
		NSMutableData *read = [NSMutableData data];
		NSInputStream *stream = self.request.HTTPBodyStream;
		[stream open];
		uint8_t buffer[4096];
		NSInteger count;
		while ((count = [stream read:buffer maxLength:sizeof buffer]) > 0) {
			[read appendBytes:buffer length:(NSUInteger)count];
		}
		[stream close];
		body = read;
	}
	@synchronized (NFKMCPEventStreamProtocol.class) {
		[NFKMCPRecordedRequests addObject:self.request];
	}
	NSDictionary *message = [NSJSONSerialization JSONObjectWithData:body options:0 error:NULL];
	NSString *method = message[@"method"];
	NSMutableDictionary *headers = [NSMutableDictionary dictionaryWithObject:@"text/event-stream" forKey:@"Content-Type"];
	NSData *reply = [NSData data];
	NSInteger status = 200;
	if ([method isEqualToString:@"initialize"]) {
		headers[@"Mcp-Session-Id"] = @"session-42";
		reply = [self eventsWith:@{ @"protocolVersion": @"2025-06-18", @"capabilities": @{ @"tools": @{} },
									@"serverInfo": @{ @"name": @"stub", @"version": @"1" } } identifier:message[@"id"]];
	} else if ([method isEqualToString:@"tools/list"]) {
		reply = [self eventsWith:@{ @"tools": @[ @{ @"name": @"lookup", @"description": @"Looks up a word.",
											 @"inputSchema": @{ @"type": @"object" } } ] } identifier:message[@"id"]];
	} else {
		status = 202;
		headers = [NSMutableDictionary dictionary];
	}
	NSHTTPURLResponse *response = [[NSHTTPURLResponse alloc] initWithURL:self.request.URL statusCode:status HTTPVersion:@"HTTP/1.1" headerFields:headers];
	[self.client URLProtocol:self didReceiveResponse:response cacheStoragePolicy:NSURLCacheStorageNotAllowed];
	[self.client URLProtocol:self didLoadData:reply];
	[self.client URLProtocolDidFinishLoading:self];
}

// A progress notification first, then the response written across several data lines, broken
// where JSON takes a newline.
- (NSData *)eventsWith:(NSDictionary *)result identifier:(id)identifier
{
	NSString *progress = @"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{\"progress\":1}}";
	NSData *json = [NSJSONSerialization dataWithJSONObject:@{ @"jsonrpc": @"2.0", @"id": identifier, @"result": result }
												   options:NSJSONWritingPrettyPrinted error:NULL];
	NSString *text = [[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding];
	NSMutableString *events = [NSMutableString stringWithFormat:@"event: message\ndata: %@\n\n", progress];
	for (NSString *line in [text componentsSeparatedByString:@"\n"]) {
		[events appendFormat:@"data: %@\n", line];
	}
	[events appendString:@"\n"];
	return [events dataUsingEncoding:NSUTF8StringEncoding];
}

- (void)stopLoading {}

@end

@interface NFKMCPClientTests : XCTestCase
@property (nonatomic, strong) NFKInferenceServer *server;
@end

@implementation NFKMCPClientTests

- (void)setUp
{
	[super setUp];
	self.server = [[NFKInferenceServer alloc] init];
	self.server.port = 0;
	self.server.loopbackOnly = YES;
	self.server.apiKey = @"secret";
	self.server.requiresAPIKeyOnLoopback = YES;
	NSError *error = nil;
	XCTAssertTrue([self.server startWithError:&error], @"%@", error);
}

- (void)tearDown
{
	[self.server stop];
	[super tearDown];
}

- (NFKMCPClient *)client
{
	return [[NFKMCPClient alloc] initWithEndpointURL:self.server.localMCPURL apiKey:@"secret"];
}

- (void)testTheClientReachesAServersToolsAsInferenceTools
{
	self.server.MCPInstructions = @"Use get_weather for weather.";
	[self.server addTool:[NFKInferenceTool toolWithName:@"get_weather" description:@"Current weather in a city."
											 parameters:@{ @"type": @"object", @"properties": @{ @"city": @{ @"type": @"string" } }, @"required": @[ @"city" ] }
												handler:^NFKToolResult *(NSDictionary *arguments, NSError **error) {
		return [NFKToolResult resultWithText:[NSString stringWithFormat:@"18 C in %@", arguments[@"city"]]];
	}]];
	NFKMCPClient *client = self.client;
	NSError *error = nil;
	NSArray<NFKInferenceTool *> *tools = [client toolsWithError:&error];
	XCTAssertEqual(tools.count, 1u, @"%@", error);
	XCTAssertEqualObjects(tools[0].name, @"get_weather");
	XCTAssertEqualObjects(tools[0].toolDescription, @"Current weather in a city.");
	XCTAssertEqualObjects(tools[0].parameters[@"required"], @[ @"city" ]);
	XCTAssertEqualObjects(client.serverInfo[@"name"], @"InferKit");
	XCTAssertEqualObjects(client.instructions, @"Use get_weather for weather.");
	XCTAssertEqualObjects(client.protocolVersion, @"2025-11-25");

	NFKToolResult *result = [tools[0] invokeWithArguments:@{ @"city": @"Paris" }];
	XCTAssertFalse(result.isError, @"%@", result.text);
	XCTAssertEqualObjects(result.text, @"18 C in Paris");
	XCTAssertTrue([tools[0] invokeWithArguments:@{}].isError, @"the schema check runs before the remote call");
}

- (void)testARemoteToolsMediaArrivesAsTheContractsTypes
{
	NSURL *clip = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:[NSUUID.UUID.UUIDString stringByAppendingString:@".wav"]];
	NSData *wav = [@"RIFF....WAVEfmt " dataUsingEncoding:NSUTF8StringEncoding];
	[wav writeToURL:clip atomically:YES];
	CVPixelBufferRef square = NULL;
	CVPixelBufferCreate(kCFAllocatorDefault, 2, 2, kCVPixelFormatType_32BGRA,
						(__bridge CFDictionaryRef)@{ (id)kCVPixelBufferIOSurfacePropertiesKey: @{} }, &square);
	NSData *png = [NFKImageCoding PNGDataForImage:(__bridge id)square];
	CVPixelBufferRelease(square);
	[self addTeardownBlock:^{
		[NSFileManager.defaultManager removeItemAtURL:clip error:NULL];
	}];
	[self.server addTool:[NFKInferenceTool toolWithName:@"draw_and_speak" description:@"Makes media." parameters:@{ @"type": @"object" }
												handler:^NFKToolResult *(NSDictionary *arguments, NSError **error) {
		return [NFKToolResult resultWithText:@"Made a picture and a sound." outputs:@{ NFKOutputAudio: [NFKAudioAsset audioAssetWithFileURL:clip] }
								   imagePNGs:@[ png ]];
	}]];
	NSError *error = nil;
	NFKToolResult *result = [self.client resultOfToolNamed:@"draw_and_speak" arguments:@{} error:&error];
	XCTAssertEqualObjects(result.text, @"Made a picture and a sound.", @"%@", error);
	XCTAssertEqualObjects(result.imagePNGs, @[ png ], @"the image is shown to the calling model as it was");
	CVPixelBufferRef image = (__bridge CVPixelBufferRef)[result.outputs objectForKey:NFKOutputImage];
	XCTAssertEqual(CVPixelBufferGetWidth(image), 2u);
	NFKAudioAsset *audio = result.outputs[NFKOutputAudio];
	XCTAssertEqualObjects(audio.fileURL.pathExtension, @"wav");
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:audio.fileURL], wav);
	[NSFileManager.defaultManager removeItemAtURL:audio.fileURL error:NULL];
}

- (void)testAToolLoopCallsARemoteServersTool
{
	NFKBackendTool *echo = [NFKBackendTool toolWithBackend:[[NFKPassthroughBackend alloc] init] name:@"echo"
											outputModality:NFKModalityText purpose:@"the user asks for an echo"];
	echo.resultDescriber = ^NFKToolResult *(NFKInferenceResult *result) {
		return [NFKToolResult resultWithText:@"18 C and clear"];
	};
	[self.server addTool:echo];
	NSError *error = nil;
	NSArray<NFKInferenceTool *> *remote = [self.client toolsWithError:&error];
	XCTAssertNotNil(remote, @"%@", error);
	XCTAssertTrue([remote.firstObject.toolDescription containsString:@"Use it when the user asks for an echo."],
				  @"the server lists the default description an NFKBackendTool writes");

	NFKToolLoop *loop = [[NFKToolLoop alloc] initWithBackend:[[NFKMCPCallingModel alloc] init] tools:remote];
	NFKInferenceResult *result = [loop runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Weather?" }
																						 parameters:nil outputModality:NFKModalityText]
														error:&error];
	XCTAssertEqualObjects(result.text, @"Remote says: 18 C and clear", @"%@", error);
}

- (void)testAServersErrorsReachTheCaller
{
	NSError *error = nil;
	XCTAssertNil([self.client resultOfToolNamed:@"missing" arguments:@{} error:&error]);
	XCTAssertTrue([error.localizedDescription containsString:@"no tool named missing"], @"%@", error);

	NFKMCPClient *unauthorized = [[NFKMCPClient alloc] initWithEndpointURL:self.server.localMCPURL apiKey:@"wrong"];
	XCTAssertFalse([unauthorized connectWithError:&error]);
	XCTAssertNotNil(error);

	[self.server addTool:[NFKInferenceTool toolWithName:@"brief" description:@"d" parameters:@{ @"type": @"object" }
												handler:^NFKToolResult *(NSDictionary *arguments, NSError **handlerError) { return [NFKToolResult resultWithText:@"ok"]; }]];
	NFKInferenceTool *proxy = [[self.client toolsWithError:NULL] firstObject];
	[self.server removeToolNamed:@"brief"];
	NFKToolResult *gone = [proxy invokeWithArguments:@{}];
	XCTAssertTrue(gone.isError, @"a tool the server no longer serves fails as an error result the model reads");
}

- (void)testAnEventStreamReplyAndASessionAreRead
{
	NFKMCPEventStreamProtocol.requests = [NSMutableArray array];
	[NSURLProtocol registerClass:NFKMCPEventStreamProtocol.class];
	[self addTeardownBlock:^{
		[NSURLProtocol unregisterClass:NFKMCPEventStreamProtocol.class];
	}];
	NFKMCPClient *client = [[NFKMCPClient alloc] initWithEndpointURL:[NSURL URLWithString:@"https://mcp.stub.invalid/mcp"] apiKey:@"k"];
	NSError *error = nil;
	NSArray<NFKInferenceTool *> *tools = [client toolsWithError:&error];
	XCTAssertEqualObjects(tools.firstObject.name, @"lookup", @"%@", error);
	XCTAssertEqualObjects(client.protocolVersion, @"2025-06-18");

	NSArray<NSURLRequest *> *sent = NFKMCPEventStreamProtocol.requests;
	XCTAssertEqual(sent.count, 3u, @"initialize, the initialized notification, and tools/list");
	XCTAssertNil([sent[0] valueForHTTPHeaderField:@"Mcp-Session-Id"]);
	XCTAssertEqualObjects([sent[0] valueForHTTPHeaderField:@"Authorization"], @"Bearer k");
	XCTAssertEqualObjects([sent[2] valueForHTTPHeaderField:@"Mcp-Session-Id"], @"session-42", @"the session the server assigned goes back");
	XCTAssertEqualObjects([sent[2] valueForHTTPHeaderField:@"MCP-Protocol-Version"], @"2025-06-18");
}

@end
