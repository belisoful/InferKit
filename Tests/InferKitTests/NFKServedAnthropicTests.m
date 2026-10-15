//
//  NFKServedAnthropicTests.m
//  InferKitTests
//
//  Anthropic's Messages API served by NFKInferenceServer, over real HTTP on a loopback port: the
//  request in the contract's shape, the Message object and its event stream, tools both ways, images,
//  errors in Anthropic's shape, the x-api-key header, token counting, and NFKAnthropicBackend pointed
//  at the server.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

/*! A synchronous backend whose reply a test writes, recording what reached it, and counting tokens
	when a count is set. */
@interface NFKAnthropicRouteBackend : NSObject <NFKInferenceBackend>
@property (atomic, strong, nullable) NFKInferenceRequest *lastRequest;
@property (nonatomic, copy, nullable) NFKInferenceResult * _Nullable (^reply)(NFKInferenceRequest *request, NSError **error);
@end

@implementation NFKAnthropicRouteBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"anthropic-route-test";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	self.lastRequest = request;
	return self.reply != nil ? self.reply(request, error) : [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"ok" }];
}

@end

/*! A backend that counts a request's tokens as the number of characters in its messages' text. */
@interface NFKAnthropicRouteCountingBackend : NFKAnthropicRouteBackend
@end

@implementation NFKAnthropicRouteCountingBackend

- (nullable NSNumber *)tokenCountForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	self.lastRequest = request;
	NSUInteger count = 0;
	for (NSDictionary *message in request.messages) {
		count += [message[@"content"] isKindOfClass:NSString.class] ? [message[@"content"] length] : 0;
	}
	return @(count);
}

@end

/*! A backend whose job streams cumulative reasoning, then cumulative text, then finishes with the
	final result a test writes. */
@interface NFKAnthropicRouteStreamingBackend : NFKAnthropicRouteBackend
@property (nonatomic, copy) NSArray<NFKInferenceResult *> *partials;
@property (nonatomic, strong, nullable) NFKInferenceResult *finalResult;
@property (nonatomic, strong, nullable) NSError *finalError;
@end

@implementation NFKAnthropicRouteStreamingBackend

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	self.lastRequest = request;
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	NSArray<NFKInferenceResult *> *partials = self.partials;
	NFKInferenceResult *finalResult = self.finalResult;
	NSError *finalError = self.finalError;
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		for (NFKInferenceResult *partial in partials) {
			[job reportProgress:-1.0 partialResult:partial];
			[NSThread sleepForTimeInterval:0.02];
		}
		if (finalError != nil) {
			[job finishWithError:finalError];
		} else {
			[job finishWithResult:finalResult];
		}
	});
	return job;
}

@end

/*! One server-sent event as the test reads it back. */
@interface NFKAnthropicRouteEvent : NSObject
@property (nonatomic, copy) NSString *name;
@property (nonatomic, copy) NSDictionary *data;
@end

@implementation NFKAnthropicRouteEvent
@end

@interface NFKServedAnthropicTests : XCTestCase
@property (nonatomic, assign) NSUInteger savedRetryAttempts;
@end

@implementation NFKServedAnthropicTests

- (void)setUp
{
	[super setUp];
	self.savedRetryAttempts = NFKRemoteTransport.retryAttempts;
	NFKRemoteTransport.retryAttempts = 0;
}

- (void)tearDown
{
	NFKRemoteTransport.retryAttempts = self.savedRetryAttempts;
	[super tearDown];
}

#pragma mark Helpers

- (NFKInferenceServer *)startedServerHosting:(NSDictionary<NSString *, id<NFKInferenceBackend>> *)backends
							  configuration:(nullable void (^)(NFKInferenceServer *server))configuration
{
	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.port = 0;
	server.loopbackOnly = YES;
	for (NSString *name in backends) {
		[server addBackend:backends[name] forModelName:name];
	}
	if (configuration != nil) {
		configuration(server);
	}
	NSError *error = nil;
	XCTAssertTrue([server startWithError:&error], @"%@", error);
	[self addTeardownBlock:^{
		[server stop];
	}];
	return server;
}

- (NFKInferenceServer *)startedServerHosting:(NSDictionary<NSString *, id<NFKInferenceBackend>> *)backends
{
	return [self startedServerHosting:backends configuration:nil];
}

- (NFKAnthropicBackend *)clientFor:(NFKInferenceServer *)server model:(NSString *)model
{
	NFKAnthropicBackend *client = [NFKAnthropicBackend backendWithEndpointURL:[server.localBaseURL URLByAppendingPathComponent:@"messages"]];
	client.modelName = model;
	return client;
}

/*! The reply to a request on the server's port, its status, and its body as text. */
- (nullable NSString *)send:(NSString *)method
					   path:(NSString *)path
					   body:(nullable id)body
					headers:(nullable NSDictionary<NSString *, NSString *> *)headers
					 server:(NFKInferenceServer *)server
					 status:(NSInteger *)outStatus
{
	NSURL *url = [NSURL URLWithString:[NSString stringWithFormat:@"http://localhost:%u%@", (unsigned)server.listeningPort, path]];
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.HTTPMethod = method;
	if (body != nil) {
		request.HTTPBody = [body isKindOfClass:NSData.class] ? body : [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL];
		[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	}
	[request setValue:@"2023-06-01" forHTTPHeaderField:@"anthropic-version"];
	for (NSString *name in headers) {
		[request setValue:headers[name] forHTTPHeaderField:name];
	}
	NSHTTPURLResponse *response = nil;
	NSData *data = [NFKRemoteTransport sendRequest:request session:NSURLSession.sharedSession response:&response error:NULL];
	*outStatus = response.statusCode;
	return data != nil ? [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] : nil;
}

- (nullable NSDictionary *)postMessages:(NSDictionary *)body server:(NFKInferenceServer *)server status:(NSInteger *)outStatus
{
	NSString *text = [self send:@"POST" path:@"/v1/messages" body:body headers:nil server:server status:outStatus];
	id reply = text != nil ? [NSJSONSerialization JSONObjectWithData:[text dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL] : nil;
	return [reply isKindOfClass:NSDictionary.class] ? reply : nil;
}

/*! The events of a server-sent event stream, each event's name checked against its data's type. */
- (NSArray<NFKAnthropicRouteEvent *> *)eventsIn:(NSString *)stream
{
	NSMutableArray<NFKAnthropicRouteEvent *> *events = [NSMutableArray array];
	for (NSString *block in [stream componentsSeparatedByString:@"\n\n"]) {
		NFKAnthropicRouteEvent *event = [[NFKAnthropicRouteEvent alloc] init];
		for (NSString *line in [block componentsSeparatedByString:@"\n"]) {
			if ([line hasPrefix:@"event: "]) {
				event.name = [line substringFromIndex:7];
			} else if ([line hasPrefix:@"data: "]) {
				event.data = [NSJSONSerialization JSONObjectWithData:[[line substringFromIndex:6] dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
			}
		}
		if (event.name != nil) {
			XCTAssertEqualObjects(event.data[@"type"], event.name, @"the event line names the data's type");
			[events addObject:event];
		}
	}
	return events;
}

- (NSString *)base64PNGWidth:(size_t)width height:(size_t)height
{
	CVPixelBufferRef buffer = NULL;
	NSDictionary *attributes = @{ (id)kCVPixelBufferIOSurfacePropertiesKey: @{} };
	CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef)attributes, &buffer);
	CVPixelBufferLockBaseAddress(buffer, 0);
	memset(CVPixelBufferGetBaseAddress(buffer), 0x7F, CVPixelBufferGetBytesPerRow(buffer) * height);
	CVPixelBufferUnlockBaseAddress(buffer, 0);
	NSData *png = [NFKImageCoding PNGDataForImage:(__bridge id)buffer];
	CVPixelBufferRelease(buffer);
	return [png base64EncodedStringWithOptions:0];
}

#pragma mark Requests

- (void)testAMessagesBodyReachesTheBackendInTheContractShape
{
	NFKAnthropicRouteBackend *backend = [[NFKAnthropicRouteBackend alloc] init];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend }];
	NSDictionary *weather = @{ @"name": @"get_weather", @"description": @"The weather in a city.",
							   @"input_schema": @{ @"type": @"object", @"properties": @{ @"city": @{ @"type": @"string" } } },
							   @"cache_control": @{ @"type": @"ephemeral" } };
	NSDictionary *body = @{ @"model": @"served", @"max_tokens": @256, @"temperature": @0.3, @"top_p": @0.9, @"top_k": @40,
							@"stop_sequences": @[ @"END" ], @"metadata": @{ @"user_id": @"someone" },
							@"system": @[ @{ @"type": @"text", @"text": @"You are terse." }, @{ @"type": @"text", @"text": @"Answer in English." } ],
							@"tools": @[ weather, @{ @"type": @"web_search_20250305", @"name": @"web_search" } ],
							@"tool_choice": @{ @"type": @"any", @"disable_parallel_tool_use": @YES },
							@"thinking": @{ @"type": @"adaptive" }, @"output_config": @{ @"effort": @"medium" },
							@"messages": @[ @{ @"role": @"user", @"content": @"Weather in Paris?" },
											@{ @"role": @"assistant", @"content": @[ @{ @"type": @"thinking", @"thinking": @"Look it up.", @"signature": @"sig" },
																					   @{ @"type": @"text", @"text": @"Checking." },
																					   @{ @"type": @"tool_use", @"id": @"toolu_1", @"name": @"get_weather",
																						  @"input": @{ @"city": @"Paris" } } ] },
											@{ @"role": @"user", @"content": @[ @{ @"type": @"tool_result", @"tool_use_id": @"toolu_1", @"is_error": @YES,
																				  @"content": @[ @{ @"type": @"text", @"text": @"service down" },
																								 @{ @"type": @"image", @"source": @{ @"type": @"base64", @"media_type": @"image/png",
																																	@"data": [self base64PNGWidth:2 height:2] } } ] },
																			   @{ @"type": @"text", @"text": @"Try again.", @"cache_control": @{ @"type": @"ephemeral" } } ] } ] };
	NSInteger status = 0;
	NSDictionary *reply = [self postMessages:body server:server status:&status];
	XCTAssertEqual(status, 200, @"%@", reply);

	NFKInferenceRequest *received = backend.lastRequest;
	NSArray<NSDictionary *> *messages = received.messages;
	XCTAssertEqual(messages.count, 5u, @"%@", messages);
	XCTAssertEqualObjects(messages[0], (@{ @"role": @"system", @"content": @"You are terse.\n\nAnswer in English." }));
	XCTAssertEqualObjects(messages[1], (@{ @"role": @"user", @"content": @"Weather in Paris?" }));
	XCTAssertEqualObjects(messages[2][@"content"], @"Checking.", @"the thinking block is not replayed");
	NSDictionary *call = [NFKToolMessages toolCallsOfAssistantMessage:messages[2]].firstObject;
	XCTAssertEqualObjects(call[@"id"], @"toolu_1");
	XCTAssertEqualObjects(call[@"name"], @"get_weather");
	XCTAssertEqualObjects(call[@"arguments"], (@{ @"city": @"Paris" }));
	XCTAssertEqualObjects(messages[3][@"role"], @"tool");
	XCTAssertEqualObjects(messages[3][@"tool_call_id"], @"toolu_1");
	XCTAssertEqualObjects([NFKToolMessages textOfToolMessage:messages[3]], @"service down");
	XCTAssertTrue([NFKToolMessages isErrorToolMessage:messages[3]]);
	XCTAssertEqual([NFKToolMessages imagePNGsOfToolMessage:messages[3]].count, 1u);
	XCTAssertEqualObjects([NFKToolMessages toolNameOfMessage:messages[3] inConversation:messages], @"get_weather");
	XCTAssertEqualObjects(messages[4], (@{ @"role": @"user", @"content": @"Try again." }));
	XCTAssertNil(received.prompt, @"a conversation of several turns is not a lone prompt");

	XCTAssertEqualObjects([received parameterForKey:NFKParameterMaxTokens], @256);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterTemperature], @0.3);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterTopP], @0.9);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterTopK], @40);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterStopSequences], (@[ @"END" ]));
	XCTAssertEqualObjects([received parameterForKey:NFKParameterReasoningEffort], NFKReasoningEffortModerate);
	NSDictionary *contractTool = @{ @"name": @"get_weather", @"description": @"The weather in a city.", @"parameters": weather[@"input_schema"] };
	XCTAssertEqualObjects([received parameterForKey:NFKParameterTools], (@[ contractTool ]), @"a server tool has no schema to offer");
	XCTAssertEqualObjects([received parameterForKey:@"tool_choice"], @"required");
	XCTAssertEqualObjects([received parameterForKey:@"parallel_tool_calls"], @NO);
	XCTAssertNil([received parameterForKey:@"metadata"]);
	XCTAssertNil([received parameterForKey:@"system"]);
	XCTAssertEqual(received.outputModality, NFKModalityText);
}

- (void)testAThinkingBudgetArrivesAsItsTokenCount
{
	NFKAnthropicRouteBackend *backend = [[NFKAnthropicRouteBackend alloc] init];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend }];
	NSInteger status = 0;
	[self postMessages:@{ @"model": @"served", @"max_tokens": @8000, @"thinking": @{ @"type": @"enabled", @"budget_tokens": @4096 },
						  @"messages": @[ @{ @"role": @"user", @"content": @"Think." } ] } server:server status:&status];
	XCTAssertEqual(status, 200);
	XCTAssertEqualObjects([backend.lastRequest parameterForKey:NFKParameterReasoningEffort], @"4096");
	XCTAssertEqualObjects(backend.lastRequest.prompt, @"Think.", @"a lone user turn also arrives as the prompt");

	[self postMessages:@{ @"model": @"served", @"max_tokens": @8000, @"thinking": @{ @"type": @"disabled" },
						  @"messages": @[ @{ @"role": @"user", @"content": @"Answer." } ] } server:server status:&status];
	XCTAssertNil([backend.lastRequest parameterForKey:NFKParameterReasoningEffort]);
}

#pragma mark Replies

- (void)testAReplyIsAMessageWithItsUsageAndStopReason
{
	NFKAnthropicRouteBackend *backend = [[NFKAnthropicRouteBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"Sunny.",
														NFKOutputUsage: @{ NFKUsageInputTokens: @30, NFKUsageCachedTokens: @12, NFKUsageOutputTokens: @3 } }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend }];
	NSInteger status = 0;
	NSDictionary *reply = [self postMessages:@{ @"model": @"served", @"max_tokens": @64,
												@"messages": @[ @{ @"role": @"user", @"content": @"Weather?" } ] } server:server status:&status];
	XCTAssertEqual(status, 200);
	XCTAssertTrue([reply[@"id"] hasPrefix:@"msg_"], @"%@", reply);
	XCTAssertEqualObjects(reply[@"type"], @"message");
	XCTAssertEqualObjects(reply[@"role"], @"assistant");
	XCTAssertEqualObjects(reply[@"model"], @"served");
	XCTAssertEqualObjects(reply[@"content"], (@[ @{ @"type": @"text", @"text": @"Sunny." } ]));
	XCTAssertEqualObjects(reply[@"stop_reason"], @"end_turn");
	XCTAssertEqualObjects(reply[@"stop_sequence"], NSNull.null);
	XCTAssertEqualObjects(reply[@"usage"], (@{ @"input_tokens": @18, @"cache_read_input_tokens": @12, @"output_tokens": @3 }),
						  @"input_tokens excludes what the cache served, as the API counts it");
}

- (void)testToolCallsComeBackAsToolUseBlocksAfterTheThinking
{
	NFKAnthropicRouteBackend *backend = [[NFKAnthropicRouteBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputReasoning: @"I need the weather.",
														NFKOutputToolCalls: @[ @{ @"id": @"call_9", @"name": @"get_weather", @"arguments": @{ @"city": @"Oslo" } },
																			   @{ @"name": @"get_time", @"argumentsJSON": @"{\"zone\":\"CET\"}" } ] }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend }];
	NSInteger status = 0;
	NSDictionary *reply = [self postMessages:@{ @"model": @"served", @"max_tokens": @64,
												@"messages": @[ @{ @"role": @"user", @"content": @"Weather and time in Oslo?" } ] } server:server status:&status];
	XCTAssertEqual(status, 200);
	NSArray *content = reply[@"content"];
	XCTAssertEqual(content.count, 3u, @"%@", content);
	XCTAssertEqualObjects(content[0], (@{ @"type": @"thinking", @"thinking": @"I need the weather.", @"signature": @"" }));
	XCTAssertEqualObjects(content[1], (@{ @"type": @"tool_use", @"id": @"call_9", @"name": @"get_weather", @"input": @{ @"city": @"Oslo" } }));
	XCTAssertEqualObjects(content[2][@"name"], @"get_time");
	XCTAssertEqualObjects(content[2][@"input"], (@{ @"zone": @"CET" }), @"arguments given only as JSON text are parsed");
	XCTAssertTrue([content[2][@"id"] hasPrefix:@"toolu_"], @"a call without an id is given one");
	XCTAssertEqualObjects(reply[@"stop_reason"], @"tool_use");
	XCTAssertEqualObjects(reply[@"usage"], (@{ @"input_tokens": @0, @"output_tokens": @0 }), @"every reply carries usage");
}

- (void)testReachingMaxTokensIsTheStopReason
{
	NFKAnthropicRouteBackend *backend = [[NFKAnthropicRouteBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"Once upon",
														NFKOutputUsage: @{ NFKUsageInputTokens: @5, NFKUsageOutputTokens: @2 } }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend }];
	NSInteger status = 0;
	NSDictionary *reply = [self postMessages:@{ @"model": @"served", @"max_tokens": @2,
												@"messages": @[ @{ @"role": @"user", @"content": @"A story." } ] } server:server status:&status];
	XCTAssertEqualObjects(reply[@"stop_reason"], @"max_tokens");
}

- (void)testAForcedToolIsStructuredOutput
{
	NFKAnthropicRouteBackend *backend = [[NFKAnthropicRouteBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputStructured: @{ @"city": @"Rome", @"celsius": @21 } }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend }];
	NSDictionary *schema = @{ @"type": @"object", @"properties": @{ @"city": @{ @"type": @"string" }, @"celsius": @{ @"type": @"number" } } };
	NSDictionary *body = @{ @"model": @"served", @"max_tokens": @64,
							@"tools": @[ @{ @"name": @"record", @"input_schema": schema }, @{ @"name": @"other", @"input_schema": @{ @"type": @"object" } } ],
							@"tool_choice": @{ @"type": @"tool", @"name": @"record" },
							@"messages": @[ @{ @"role": @"user", @"content": @"Rome is 21 degrees." } ] };
	NSInteger status = 0;
	NSDictionary *reply = [self postMessages:body server:server status:&status];
	XCTAssertEqual(status, 200, @"%@", reply);
	XCTAssertEqualObjects([backend.lastRequest parameterForKey:NFKParameterJSONSchema], schema);
	XCTAssertNil([backend.lastRequest parameterForKey:NFKParameterTools], @"a forced tool leaves no other tool to call");
	NSDictionary *use = [reply[@"content"] firstObject];
	XCTAssertEqualObjects(use[@"type"], @"tool_use");
	XCTAssertEqualObjects(use[@"name"], @"record");
	XCTAssertEqualObjects(use[@"input"], (@{ @"city": @"Rome", @"celsius": @21 }));
	XCTAssertEqual([reply[@"content"] count], 1u);
	XCTAssertEqualObjects(reply[@"stop_reason"], @"tool_use");

	NSMutableDictionary *undeclared = [body mutableCopy];
	undeclared[@"tool_choice"] = @{ @"type": @"tool", @"name": @"absent" };
	reply = [self postMessages:undeclared server:server status:&status];
	XCTAssertEqual(status, 400);
	XCTAssertEqualObjects(reply[@"error"][@"type"], @"invalid_request_error");
}

#pragma mark Images

- (void)testAnImageBlockArrivesAsAPixelBufferAndAURLIsRefused
{
	NFKAnthropicRouteBackend *backend = [[NFKAnthropicRouteBackend alloc] init];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"vision": backend }];
	NSDictionary *image = @{ @"type": @"image", @"source": @{ @"type": @"base64", @"media_type": @"image/png", @"data": [self base64PNGWidth:5 height:3] } };
	NSInteger status = 0;
	[self postMessages:@{ @"model": @"vision", @"max_tokens": @16,
						  @"messages": @[ @{ @"role": @"user", @"content": @[ image, image, @{ @"type": @"text", @"text": @"Compare." } ] } ] }
				server:server status:&status];
	XCTAssertEqual(status, 200);
	CVPixelBufferRef received = (__bridge CVPixelBufferRef)[backend.lastRequest inputForKey:NFKInputImage];
	XCTAssertTrue(received != NULL && CFGetTypeID(received) == CVPixelBufferGetTypeID());
	XCTAssertEqual(CVPixelBufferGetWidth(received), 5u);
	XCTAssertEqual(CVPixelBufferGetHeight(received), 3u);
	XCTAssertEqual([[backend.lastRequest inputForKey:NFKInputImages] count], 1u, @"the second image rides under NFKInputImages");
	XCTAssertEqualObjects(backend.lastRequest.prompt, @"Compare.");

	NSDictionary *linked = @{ @"type": @"image", @"source": @{ @"type": @"url", @"url": @"https://example.com/a.png" } };
	NSDictionary *reply = [self postMessages:@{ @"model": @"vision", @"max_tokens": @16,
												@"messages": @[ @{ @"role": @"user", @"content": @[ linked ] } ] } server:server status:&status];
	XCTAssertEqual(status, 400);
	XCTAssertEqualObjects(reply[@"type"], @"error");
	XCTAssertEqualObjects(reply[@"error"][@"type"], @"invalid_request_error");
	XCTAssertTrue([reply[@"error"][@"message"] containsString:@"base64"], @"%@", reply);
}

#pragma mark Streaming

- (void)testAStreamSendsTheMessagesEventsInOrder
{
	NFKAnthropicRouteStreamingBackend *backend = [[NFKAnthropicRouteStreamingBackend alloc] init];
	backend.partials = @[ [NFKInferenceResult resultWithOutputs:@{ NFKOutputReasoning: @"Hmm" }],
						  [NFKInferenceResult resultWithOutputs:@{ NFKOutputReasoning: @"Hmm, Oslo." }],
						  [NFKInferenceResult resultWithOutputs:@{ NFKOutputReasoning: @"Hmm, Oslo.", NFKOutputText: @"Let me" }],
						  [NFKInferenceResult resultWithOutputs:@{ NFKOutputReasoning: @"Hmm, Oslo.", NFKOutputText: @"Let me check." }] ];
	backend.finalResult = [NFKInferenceResult resultWithOutputs:@{ NFKOutputReasoning: @"Hmm, Oslo.", NFKOutputText: @"Let me check.",
																   NFKOutputToolCalls: @[ @{ @"id": @"toolu_7", @"name": @"get_weather",
																							 @"arguments": @{ @"city": @"Oslo" } } ],
																   NFKOutputUsage: @{ NFKUsageInputTokens: @9, NFKUsageOutputTokens: @6 } }];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"streamer": backend }];
	NSInteger status = 0;
	NSString *stream = [self send:@"POST" path:@"/v1/messages"
							 body:@{ @"model": @"streamer", @"max_tokens": @64, @"stream": @YES,
									 @"messages": @[ @{ @"role": @"user", @"content": @"Weather in Oslo?" } ] }
						  headers:nil server:server status:&status];
	XCTAssertEqual(status, 200);
	NSArray<NFKAnthropicRouteEvent *> *events = [self eventsIn:stream];
	NSArray<NSString *> *names = [events valueForKey:@"name"];
	NSArray<NSString *> *expected = @[ @"message_start",
									   @"content_block_start", @"content_block_delta", @"content_block_delta", @"content_block_stop",
									   @"content_block_start", @"content_block_delta", @"content_block_delta", @"content_block_stop",
									   @"content_block_start", @"content_block_delta", @"content_block_stop",
									   @"message_delta", @"message_stop" ];
	XCTAssertEqualObjects(names, expected, @"%@", stream);
	if (![names isEqualToArray:expected]) {
		return;
	}
	XCTAssertTrue([events[0].data[@"message"][@"id"] hasPrefix:@"msg_"]);
	XCTAssertEqualObjects(events[0].data[@"message"][@"content"], @[]);
	XCTAssertEqualObjects(events[1].data[@"content_block"][@"type"], @"thinking");
	XCTAssertEqualObjects(events[2].data[@"delta"], (@{ @"type": @"thinking_delta", @"thinking": @"Hmm" }));
	XCTAssertEqualObjects(events[3].data[@"delta"], (@{ @"type": @"thinking_delta", @"thinking": @", Oslo." }));
	XCTAssertEqualObjects(events[5].data[@"index"], @1);
	XCTAssertEqualObjects(events[5].data[@"content_block"], (@{ @"type": @"text", @"text": @"" }));
	XCTAssertEqualObjects(events[6].data[@"delta"], (@{ @"type": @"text_delta", @"text": @"Let me" }));
	XCTAssertEqualObjects(events[7].data[@"delta"], (@{ @"type": @"text_delta", @"text": @" check." }));
	XCTAssertEqualObjects(events[9].data[@"content_block"], (@{ @"type": @"tool_use", @"id": @"toolu_7", @"name": @"get_weather", @"input": @{} }));
	XCTAssertEqualObjects(events[10].data[@"delta"], (@{ @"type": @"input_json_delta", @"partial_json": @"{\"city\":\"Oslo\"}" }));
	XCTAssertEqualObjects(events[11].data[@"index"], @2);
	XCTAssertEqualObjects(events[12].data[@"delta"][@"stop_reason"], @"tool_use");
	XCTAssertEqualObjects(events[12].data[@"usage"], (@{ @"input_tokens": @9, @"output_tokens": @6 }));
}

- (void)testASynchronousBackendStreamsItsWholeReplyAtTheEnd
{
	NFKAnthropicRouteBackend *backend = [[NFKAnthropicRouteBackend alloc] init];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend }];
	NSInteger status = 0;
	NSString *stream = [self send:@"POST" path:@"/v1/messages"
							 body:@{ @"model": @"served", @"max_tokens": @8, @"stream": @YES, @"messages": @[ @{ @"role": @"user", @"content": @"Hi" } ] }
						  headers:nil server:server status:&status];
	NSArray<NFKAnthropicRouteEvent *> *events = [self eventsIn:stream];
	XCTAssertEqualObjects([events valueForKey:@"name"], (@[ @"message_start", @"content_block_start", @"content_block_delta",
															 @"content_block_stop", @"message_delta", @"message_stop" ]), @"%@", stream);
	XCTAssertEqualObjects(events[2].data[@"delta"][@"text"], @"ok");
	XCTAssertEqualObjects(events[4].data[@"delta"][@"stop_reason"], @"end_turn");
}

- (void)testAFailureAfterTheStreamOpensIsAnErrorEvent
{
	NFKAnthropicRouteStreamingBackend *backend = [[NFKAnthropicRouteStreamingBackend alloc] init];
	backend.partials = @[ [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"partial" }] ];
	backend.finalError = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceRateLimited
										 userInfo:@{ NSLocalizedDescriptionKey: @"out of quota" }];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"flaky": backend }];
	NSInteger status = 0;
	NSString *stream = [self send:@"POST" path:@"/v1/messages"
							 body:@{ @"model": @"flaky", @"max_tokens": @8, @"stream": @YES, @"messages": @[ @{ @"role": @"user", @"content": @"Go" } ] }
						  headers:nil server:server status:&status];
	XCTAssertEqual(status, 200);
	NFKAnthropicRouteEvent *last = [self eventsIn:stream].lastObject;
	XCTAssertEqualObjects(last.name, @"error", @"%@", stream);
	XCTAssertEqualObjects(last.data[@"error"][@"type"], @"rate_limit_error");
	XCTAssertEqualObjects(last.data[@"error"][@"message"], @"out of quota");

	backend.partials = @[];
	stream = [self send:@"POST" path:@"/v1/messages"
				   body:@{ @"model": @"flaky", @"max_tokens": @8, @"stream": @YES, @"messages": @[ @{ @"role": @"user", @"content": @"Go" } ] }
				headers:nil server:server status:&status];
	XCTAssertEqual(status, 429, @"a failure before the first event is the reply's status");
	XCTAssertTrue([stream containsString:@"\"rate_limit_error\""], @"%@", stream);
}

#pragma mark Errors

- (void)testErrorsTakeAnthropicsShapeAndStatus
{
	NFKAnthropicRouteBackend *backend = [[NFKAnthropicRouteBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceNotReady userInfo:@{ NSLocalizedDescriptionKey: @"still loading" }];
		return nil;
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"a": backend, @"b": [[NFKAnthropicRouteBackend alloc] init] }];
	NSInteger status = 0;
	NSDictionary *reply = [self postMessages:@{ @"model": @"claude-sonnet-4-5", @"max_tokens": @8,
												@"messages": @[ @{ @"role": @"user", @"content": @"Hi" } ] } server:server status:&status];
	XCTAssertEqual(status, 404);
	XCTAssertEqualObjects(reply[@"type"], @"error");
	XCTAssertEqualObjects(reply[@"error"][@"type"], @"not_found_error");
	XCTAssertTrue([reply[@"error"][@"message"] containsString:@"claude-sonnet-4-5"]);

	reply = [self postMessages:@{ @"model": @"a", @"max_tokens": @8, @"messages": @[ @{ @"role": @"user", @"content": @"Hi" } ] } server:server status:&status];
	XCTAssertEqual(status, 503);
	XCTAssertEqualObjects(reply[@"error"][@"type"], @"overloaded_error");
	XCTAssertEqualObjects(reply[@"error"][@"message"], @"still loading");

	reply = [self postMessages:@{ @"model": @"b", @"max_tokens": @8, @"messages": @[] } server:server status:&status];
	XCTAssertEqual(status, 400);
	XCTAssertEqualObjects(reply[@"error"][@"type"], @"invalid_request_error");

	reply = [self postMessages:@{ @"model": @"b", @"messages": @[ @{ @"role": @"user", @"content": @[ @{ @"type": @"tool_result", @"tool_use_id": @"nowhere" } ] } ] }
						server:server status:&status];
	XCTAssertEqual(status, 400, @"a tool_result answers a tool_use in the conversation");

	NSString *text = [self send:@"POST" path:@"/v1/messages" body:[@"not json" dataUsingEncoding:NSUTF8StringEncoding] headers:nil server:server status:&status];
	XCTAssertEqual(status, 400);
	XCTAssertTrue([text containsString:@"\"invalid_request_error\""], @"%@", text);

	text = [self send:@"GET" path:@"/v1/messages" body:nil headers:nil server:server status:&status];
	XCTAssertEqual(status, 405);
	XCTAssertTrue([text containsString:@"\"type\":\"error\""], @"%@", text);

	text = [self send:@"POST" path:@"/v1/messages/batches" body:@{} headers:nil server:server status:&status];
	XCTAssertEqual(status, 404);
	XCTAssertTrue([text containsString:@"\"not_found_error\""], @"%@", text);
}

#pragma mark Keys

- (void)testTheMessagesRoutesTakeTheKeyAsXAPIKey
{
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": [[NFKAnthropicRouteBackend alloc] init] }
											  configuration:^(NFKInferenceServer *configured) {
		configured.loopbackExemptScopes = 0;
		[configured addAPIKey:@"key-inference" named:@"agent" scopes:NFKServerKeyScopeInference];
		[configured addAPIKey:@"key-status" named:@"dashboard" scopes:NFKServerKeyScopeStatus];
	}];
	NSDictionary *body = @{ @"model": @"served", @"max_tokens": @8, @"messages": @[ @{ @"role": @"user", @"content": @"Hi" } ] };
	NSInteger status = 0;
	NSString *text = [self send:@"POST" path:@"/v1/messages" body:body headers:nil server:server status:&status];
	XCTAssertEqual(status, 401);
	XCTAssertTrue([text containsString:@"\"authentication_error\""], @"%@", text);
	XCTAssertTrue([text containsString:@"x-api-key"], @"the refusal names the header the route takes: %@", text);

	[self send:@"POST" path:@"/v1/messages" body:body headers:@{ @"x-api-key": @"key-inference" } server:server status:&status];
	XCTAssertEqual(status, 200);
	[self send:@"POST" path:@"/v1/messages" body:body headers:@{ @"Authorization": @"Bearer key-inference" } server:server status:&status];
	XCTAssertEqual(status, 200, @"a Bearer token opens the route too");
	[self send:@"POST" path:@"/v1/messages" body:body headers:@{ @"x-api-key": @"key-inference", @"anthropic-version": @"2099-01-01" }
		server:server status:&status];
	XCTAssertEqual(status, 200, @"any anthropic-version is accepted");

	text = [self send:@"POST" path:@"/v1/messages" body:body headers:@{ @"x-api-key": @"key-status" } server:server status:&status];
	XCTAssertEqual(status, 403);
	XCTAssertTrue([text containsString:@"\"permission_error\""], @"%@", text);
	[self send:@"POST" path:@"/v1/messages" body:body headers:@{ @"x-api-key": @"key-wrong" } server:server status:&status];
	XCTAssertEqual(status, 401);

	[self send:@"POST" path:@"/v1/chat/completions" body:body headers:@{ @"x-api-key": @"key-inference" } server:server status:&status];
	XCTAssertEqual(status, 401, @"x-api-key opens only Anthropic's routes");
	[self send:@"POST" path:@"/v1/responses" body:@{ @"model": @"served", @"input": @"Hi" } headers:@{ @"x-api-key": @"key-inference" }
		server:server status:&status];
	XCTAssertEqual(status, 401);
}

#pragma mark Model list

- (void)testTheModelListTakesAnthropicsShapeForAnAnthropicClient
{
	NFKInferenceServer *server = [self startedServerHosting:@{ @"alpha": [[NFKAnthropicRouteBackend alloc] init],
															   @"beta": [[NFKAnthropicRouteBackend alloc] init] }
											  configuration:^(NFKInferenceServer *configured) {
		configured.loopbackExemptScopes = 0;
		[configured addAPIKey:@"key-inference" named:@"agent" scopes:NFKServerKeyScopeInference];
	}];
	NSInteger status = 0;
	NSString *text = [self send:@"GET" path:@"/v1/models" body:nil headers:@{ @"x-api-key": @"key-inference" } server:server status:&status];
	XCTAssertEqual(status, 200, @"an Anthropic client's key opens the list: %@", text);
	NSDictionary *list = [NSJSONSerialization JSONObjectWithData:[text dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
	XCTAssertEqualObjects([list[@"data"] valueForKey:@"id"], (@[ @"alpha", @"beta" ]));
	XCTAssertEqualObjects([list[@"data"] valueForKey:@"type"], (@[ @"model", @"model" ]));
	XCTAssertEqualObjects([list[@"data"] valueForKey:@"display_name"], (@[ @"alpha", @"beta" ]));
	XCTAssertEqualObjects(list[@"has_more"], @NO);
	XCTAssertEqualObjects(list[@"first_id"], @"alpha");
	XCTAssertEqualObjects(list[@"last_id"], @"beta");
	XCTAssertNil(list[@"object"]);
	NSString *created = [list[@"data"] firstObject][@"created_at"];
	XCTAssertNotNil([[[NSISO8601DateFormatter alloc] init] dateFromString:created], @"created_at is an RFC 3339 date: %@", created);

	text = [self send:@"GET" path:@"/v1/models/beta" body:nil headers:@{ @"x-api-key": @"key-inference" } server:server status:&status];
	NSDictionary *entry = [NSJSONSerialization JSONObjectWithData:[text dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
	XCTAssertEqualObjects(entry[@"type"], @"model", @"%@", text);
	XCTAssertEqualObjects(entry[@"id"], @"beta");
	text = [self send:@"GET" path:@"/v1/models/absent" body:nil headers:@{ @"x-api-key": @"key-inference" } server:server status:&status];
	XCTAssertEqual(status, 404);
	XCTAssertTrue([text containsString:@"\"not_found_error\""], @"%@", text);

	NSMutableURLRequest *openAI = [NSMutableURLRequest requestWithURL:[server.localBaseURL URLByAppendingPathComponent:@"models"]];
	[openAI setValue:@"Bearer key-inference" forHTTPHeaderField:@"Authorization"];
	NSHTTPURLResponse *response = nil;
	NSData *data = [NFKRemoteTransport sendRequest:openAI session:NSURLSession.sharedSession response:&response error:NULL];
	NSDictionary *openAIList = data != nil ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
	XCTAssertEqual(response.statusCode, 200);
	XCTAssertEqualObjects(openAIList[@"object"], @"list", @"without anthropic-version the list keeps OpenAI's shape");
	XCTAssertEqualObjects([openAIList[@"data"] firstObject][@"object"], @"model");
}

#pragma mark Token counting

- (void)testCountTokensAnswersFromABackendThatCounts
{
	NFKAnthropicRouteCountingBackend *counting = [[NFKAnthropicRouteCountingBackend alloc] init];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"counts": counting, @"silent": [[NFKAnthropicRouteBackend alloc] init] }];
	NSDictionary *body = @{ @"model": @"counts", @"system": @"Be brief.", @"messages": @[ @{ @"role": @"user", @"content": @"Hello" } ] };
	NSInteger status = 0;
	NSString *text = [self send:@"POST" path:@"/v1/messages/count_tokens" body:body headers:nil server:server status:&status];
	XCTAssertEqual(status, 200, @"%@", text);
	NSDictionary *reply = [NSJSONSerialization JSONObjectWithData:[text dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
	XCTAssertEqualObjects(reply, (@{ @"input_tokens": @14 }), @"the backend counted the system turn and the user turn");
	XCTAssertEqualObjects(counting.lastRequest.messages.firstObject[@"role"], @"system");

	NSMutableDictionary *silent = [body mutableCopy];
	silent[@"model"] = @"silent";
	text = [self send:@"POST" path:@"/v1/messages/count_tokens" body:silent headers:nil server:server status:&status];
	XCTAssertEqual(status, 400);
	XCTAssertTrue([text containsString:@"does not count tokens"], @"%@", text);
	XCTAssertTrue([text containsString:@"\"invalid_request_error\""], @"%@", text);
}

- (void)testTheAnthropicClientCountsThroughCountTokens
{
	NFKAnthropicRouteCountingBackend *counting = [[NFKAnthropicRouteCountingBackend alloc] init];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"counts": counting, @"silent": [[NFKAnthropicRouteBackend alloc] init] }];
	NFKAnthropicBackend *client = [self clientFor:server model:@"counts"];
	NSDictionary *tool = @{ @"name": @"lookup", @"parameters": @{ @"type": @"object" } };
	NSArray *messages = @[ @{ @"role": @"system", @"content": @"Be brief." }, @{ @"role": @"user", @"content": @"Hello" } ];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: messages }
															   parameters:@{ NFKParameterTools: @[ tool ], NFKParameterMaxTokens: @99 }
														   outputModality:NFKModalityText];
	NSError *error = nil;
	XCTAssertEqualObjects([client tokenCountForRequest:request error:&error], @14, @"%@", error);
	XCTAssertEqualObjects(counting.lastRequest.messages, messages, @"the count reads the conversation a run sends");
	XCTAssertEqualObjects([[counting.lastRequest parameterForKey:NFKParameterTools] firstObject][@"name"], @"lookup", @"and its tools");
	XCTAssertNil([counting.lastRequest parameterForKey:NFKParameterMaxTokens], @"count_tokens takes no max_tokens");

	client.modelName = @"silent";
	XCTAssertNil([client tokenCountForRequest:request error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported, @"%@", error);
}

- (void)testTheTokenCounterSendsWhatTheClientCounts
{
	NFKAnthropicRouteCountingBackend *counting = [[NFKAnthropicRouteCountingBackend alloc] init];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"counts": counting }];
	NFKRemoteTokenCounter *counter = [[NFKRemoteTokenCounter alloc] init];
	counter.apiStyle = NFKRemoteTokenCounterAPIStyleAnthropic;
	counter.endpointURL = [server.localBaseURL URLByAppendingPathComponent:@"messages/count_tokens"];
	counter.modelName = @"counts";
	NSDictionary *call = @{ @"id": @"call_1", @"type": @"function", @"function": @{ @"name": @"lookup", @"arguments": @"{\"term\":\"orbit\"}" } };
	NSArray *messages = @[ @{ @"role": @"system", @"content": @"Be brief." },
						   @{ @"role": @"user", @"content": @"Define orbit." },
						   @{ @"role": @"assistant", @"content": NSNull.null, @"tool_calls": @[ call ] },
						   [NFKToolMessages messageForToolCallIdentifier:@"call_1" name:@"lookup" result:[NFKToolResult resultWithText:@"a path"]] ];
	CVPixelBufferRef image = NULL;
	NSDictionary *attributes = @{ (id)kCVPixelBufferIOSurfacePropertiesKey: @{} };
	CVPixelBufferCreate(kCFAllocatorDefault, 4, 4, kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef)attributes, &image);
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: messages, NFKInputImage: (__bridge id)image }
															   parameters:@{ NFKParameterTools: @[ @{ @"name": @"lookup", @"parameters": @{ @"type": @"object" } } ] }
														   outputModality:NFKModalityText];
	CVPixelBufferRelease(image);

	NSError *error = nil;
	NSNumber *counted = [counter tokenCountForRequest:request error:&error];
	XCTAssertNotNil(counted, @"%@", error);
	NFKInferenceRequest *received = counting.lastRequest;
	XCTAssertEqualObjects([received.messages valueForKey:@"role"], (@[ @"system", @"user", @"assistant", @"tool" ]),
						  @"the tool turns arrive converted and back: %@", received.messages);
	XCTAssertEqualObjects([NFKToolMessages toolCallsOfAssistantMessage:received.messages[2]].firstObject[@"id"], @"call_1");
	XCTAssertEqualObjects([NFKToolMessages textOfToolMessage:received.messages[3]], @"a path");
	XCTAssertEqualObjects([[received parameterForKey:NFKParameterTools] firstObject][@"name"], @"lookup", @"the tools are counted");
	XCTAssertNotNil([received inputForKey:NFKInputImage], @"the media are counted");

	NFKAnthropicBackend *client = [self clientFor:server model:@"counts"];
	XCTAssertEqualObjects([client tokenCountForRequest:request error:&error], counted, @"the counter and the client count the same body: %@", error);
}

#pragma mark NFKAnthropicBackend against the server

- (void)testTheAnthropicClientRunsAHostedModelWithItsKey
{
	NFKAnthropicRouteBackend *backend = [[NFKAnthropicRouteBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"Bonjour.",
														NFKOutputUsage: @{ NFKUsageInputTokens: @11, NFKUsageOutputTokens: @2 } }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend } configuration:^(NFKInferenceServer *configured) {
		configured.loopbackExemptScopes = 0;
		[configured addAPIKey:@"sesame" named:@"client" scopes:NFKServerKeyScopeInference];
	}];
	NFKAnthropicBackend *client = [self clientFor:server model:@"served"];
	NSArray *messages = @[ @{ @"role": @"system", @"content": @"Answer in French." }, @{ @"role": @"user", @"content": @"Hello" } ];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: messages }
															   parameters:@{ NFKParameterMaxTokens: @50, NFKParameterStopSequences: @[ @"STOP" ] }
														   outputModality:NFKModalityText];
	NSError *error = nil;
	XCTAssertNil([client runInferenceForRequest:request error:&error], @"no key, no answer");
	XCTAssertEqualObjects(error.userInfo[NFKRemoteErrorStatusCodeKey], @401);

	client.apiKey = @"sesame";
	NFKInferenceResult *result = [client runInferenceForRequest:request error:&error];
	XCTAssertEqualObjects(result.text, @"Bonjour.", @"%@", error);
	XCTAssertEqualObjects([result outputForKey:NFKOutputUsage][NFKUsageInputTokens], @11);
	XCTAssertEqualObjects([result outputForKey:NFKOutputUsage][NFKUsageOutputTokens], @2);
	NFKInferenceRequest *received = backend.lastRequest;
	XCTAssertEqualObjects(received.messages, messages);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterMaxTokens], @50);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterStopSequences], (@[ @"STOP" ]));

	NFKInferenceRequest *reasoned = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Why?" }
																parameters:@{ NFKParameterReasoningEffort: NFKReasoningEffortLight }
															outputModality:NFKModalityText];
	XCTAssertEqualObjects([client runInferenceForRequest:reasoned error:&error].text, @"Bonjour.", @"%@", error);
	XCTAssertEqualObjects([backend.lastRequest parameterForKey:NFKParameterReasoningEffort], NFKReasoningEffortLight,
						  @"the client's adaptive effort reads back as the contract's level");
}

- (void)testCachedTokensRoundTripThroughTheAnthropicClient
{
	NSDictionary *usage = @{ NFKUsageInputTokens: @30, NFKUsageCachedTokens: @12, NFKUsageOutputTokens: @3 };
	NFKAnthropicRouteStreamingBackend *backend = [[NFKAnthropicRouteStreamingBackend alloc] init];
	backend.partials = @[];
	backend.finalResult = [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"Cached.", NFKOutputUsage: usage }];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend }];
	NFKAnthropicBackend *client = [self clientFor:server model:@"served"];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Again?" }];

	NSInteger status = 0;
	NSDictionary *reply = [self postMessages:@{ @"model": @"served", @"max_tokens": @16,
												@"messages": @[ @{ @"role": @"user", @"content": @"Again?" } ] } server:server status:&status];
	XCTAssertEqualObjects(reply[@"usage"][@"input_tokens"], @18, @"the route reports the uncached part, as the API does");
	XCTAssertEqualObjects(reply[@"usage"][@"cache_read_input_tokens"], @12);

	NSError *error = nil;
	NFKInferenceResult *result = [client runInferenceForRequest:request error:&error];
	XCTAssertEqualObjects([result outputForKey:NFKOutputUsage], usage, @"%@", error);

	XCTestExpectation *done = [self expectationWithDescription:@"stream"];
	NFKInferenceJob *job = [client submitInferenceJobForRequest:request];
	job.completionHandler = ^(NFKInferenceJob *finished) {
		[done fulfill];
	};
	[self waitForExpectations:@[ done ] timeout:10];
	XCTAssertEqualObjects([job.result outputForKey:NFKOutputUsage], usage, @"%@", job.error);
}

- (void)testTheAnthropicClientStreamsTheHostedJob
{
	NFKAnthropicRouteStreamingBackend *backend = [[NFKAnthropicRouteStreamingBackend alloc] init];
	backend.partials = @[ [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"Hel" }],
						  [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"Hello" }] ];
	backend.finalResult = [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"Hello, world",
																   NFKOutputUsage: @{ NFKUsageInputTokens: @4, NFKUsageOutputTokens: @3 } }];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"streamer": backend }];
	NFKAnthropicBackend *client = [self clientFor:server model:@"streamer"];
	NSMutableArray<NSString *> *seen = [NSMutableArray array];
	XCTestExpectation *done = [self expectationWithDescription:@"stream"];
	NFKInferenceJob *job = [client submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"go" }]];
	job.progressHandler = ^(NFKInferenceJob *progressed) {
		@synchronized (seen) {
			if (progressed.partialResult.text.length > 0) {
				[seen addObject:progressed.partialResult.text];
			}
		}
	};
	job.completionHandler = ^(NFKInferenceJob *finished) {
		[done fulfill];
	};
	[self waitForExpectations:@[ done ] timeout:10];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	XCTAssertEqualObjects(job.result.text, @"Hello, world");
	XCTAssertEqualObjects([job.result outputForKey:NFKOutputUsage][NFKUsageOutputTokens], @3);
	XCTAssertGreaterThan(seen.count, 1u, @"the text arrived in more than one piece");
	XCTAssertEqualObjects(seen.lastObject, @"Hello, world");
}

- (void)testTheAnthropicClientsToolConversationRoundTrips
{
	NFKAnthropicRouteBackend *backend = [[NFKAnthropicRouteBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		if ([request.messages.lastObject[@"role"] isEqual:@"tool"]) {
			return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"It is 18 degrees in Lyon." }];
		}
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputToolCalls: @[ @{ @"id": @"call_lyon", @"name": @"get_weather",
																				  @"arguments": @{ @"city": @"Lyon" } } ] }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend }];
	NFKAnthropicBackend *client = [self clientFor:server model:@"served"];
	NSDictionary *tool = @{ @"name": @"get_weather", @"description": @"The weather in a city.",
							@"parameters": @{ @"type": @"object", @"properties": @{ @"city": @{ @"type": @"string" } } } };
	NSArray *opening = @[ @{ @"role": @"user", @"content": @"Weather in Lyon?" } ];
	NSError *error = nil;
	NFKInferenceResult *first = [client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: opening }
																					   parameters:@{ NFKParameterTools: @[ tool ] }
																				   outputModality:NFKModalityText] error:&error];
	XCTAssertEqualObjects(first.toolCalls.firstObject[@"name"], @"get_weather", @"%@", error);
	XCTAssertEqualObjects(first.toolCalls.firstObject[@"arguments"], (@{ @"city": @"Lyon" }));
	XCTAssertEqualObjects(first.toolCalls.firstObject[@"id"], @"call_lyon");
	XCTAssertEqualObjects([backend.lastRequest parameterForKey:NFKParameterTools], (@[ tool ]), @"the tool arrives in the contract's shape");

	NSMutableArray *conversation = [opening mutableCopy];
	[conversation addObject:[NFKToolMessages assistantMessageForResult:first]];
	[conversation addObject:[NFKToolMessages messageForToolCallIdentifier:@"call_lyon" name:@"get_weather"
																   result:[NFKToolResult resultWithText:@"18 C"]]];
	NFKInferenceResult *second = [client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: conversation }
																						parameters:@{ NFKParameterTools: @[ tool ] }
																					outputModality:NFKModalityText] error:&error];
	XCTAssertEqualObjects(second.text, @"It is 18 degrees in Lyon.", @"%@", error);
	NSArray<NSDictionary *> *received = backend.lastRequest.messages;
	XCTAssertEqual(received.count, 3u, @"%@", received);
	XCTAssertEqualObjects([NFKToolMessages toolCallsOfAssistantMessage:received[1]].firstObject[@"id"], @"call_lyon");
	XCTAssertEqualObjects(received[2][@"tool_call_id"], @"call_lyon");
	XCTAssertEqualObjects([NFKToolMessages textOfToolMessage:received[2]], @"18 C");
	XCTAssertEqualObjects([NFKToolMessages toolNameOfMessage:received[2] inConversation:received], @"get_weather");
}

- (void)testTheAnthropicClientStreamsAToolCall
{
	NFKAnthropicRouteStreamingBackend *backend = [[NFKAnthropicRouteStreamingBackend alloc] init];
	backend.partials = @[];
	backend.finalResult = [NFKInferenceResult resultWithOutputs:@{ NFKOutputToolCalls: @[ @{ @"id": @"call_s", @"name": @"lookup",
																							 @"arguments": @{ @"term": @"orbit" } } ] }];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"streamer": backend }];
	NFKAnthropicBackend *client = [self clientFor:server model:@"streamer"];
	XCTestExpectation *done = [self expectationWithDescription:@"stream"];
	NFKInferenceJob *job = [client submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Define orbit." }
																							parameters:@{ NFKParameterTools: @[ @{ @"name": @"lookup" } ] }
																						outputModality:NFKModalityText]];
	job.completionHandler = ^(NFKInferenceJob *finished) {
		[done fulfill];
	};
	[self waitForExpectations:@[ done ] timeout:10];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	XCTAssertEqualObjects(job.result.toolCalls.firstObject[@"name"], @"lookup");
	XCTAssertEqualObjects(job.result.toolCalls.firstObject[@"arguments"], (@{ @"term": @"orbit" }));
	XCTAssertEqualObjects(job.result.toolCalls.firstObject[@"id"], @"call_s");
}

- (void)testTheAnthropicClientsSchemaComesBackStructured
{
	NFKAnthropicRouteBackend *backend = [[NFKAnthropicRouteBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"{\"answer\":42}" }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend }];
	NFKAnthropicBackend *client = [self clientFor:server model:@"served"];
	NSDictionary *schema = @{ @"type": @"object", @"properties": @{ @"answer": @{ @"type": @"integer" } } };
	NSError *error = nil;
	NFKInferenceResult *result = [client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"The answer?" }
																						parameters:@{ NFKParameterJSONSchema: schema }
																					outputModality:NFKModalityText] error:&error];
	XCTAssertEqualObjects(result.structured, (@{ @"answer": @42 }), @"%@", error);
	XCTAssertEqualObjects([backend.lastRequest parameterForKey:NFKParameterJSONSchema], schema,
						  @"the client's forced structured_output tool reads back as the schema");
}

- (void)testTheAnthropicClientsImageArrivesAsAPixelBuffer
{
	NFKAnthropicRouteBackend *backend = [[NFKAnthropicRouteBackend alloc] init];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"vision": backend }];
	NFKAnthropicBackend *client = [self clientFor:server model:@"vision"];
	CVPixelBufferRef image = NULL;
	NSDictionary *attributes = @{ (id)kCVPixelBufferIOSurfacePropertiesKey: @{} };
	CVPixelBufferCreate(kCFAllocatorDefault, 7, 4, kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef)attributes, &image);
	NSError *error = nil;
	NFKInferenceResult *result = [client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"What is it?",
																										 NFKInputImage: (__bridge id)image }] error:&error];
	CVPixelBufferRelease(image);
	XCTAssertEqualObjects(result.text, @"ok", @"%@", error);
	CVPixelBufferRef received = (__bridge CVPixelBufferRef)[backend.lastRequest inputForKey:NFKInputImage];
	XCTAssertTrue(received != NULL && CFGetTypeID(received) == CVPixelBufferGetTypeID());
	XCTAssertEqual(CVPixelBufferGetWidth(received), 7u);
	XCTAssertEqual(CVPixelBufferGetHeight(received), 4u);
	XCTAssertEqualObjects(backend.lastRequest.messages.lastObject[@"content"], @"What is it?");
}

- (void)testTheAnthropicClientReadsTheServedErrorCode
{
	NFKAnthropicRouteBackend *backend = [[NFKAnthropicRouteBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceMissingInput
								 userInfo:@{ NSLocalizedDescriptionKey: @"no prompt at all" }];
		return nil;
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"failing": backend }];
	NFKAnthropicBackend *client = [self clientFor:server model:@"failing"];
	NSError *error = nil;
	XCTAssertNil([client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput, @"the code in the Anthropic error body outranks the 400: %@", error);
	XCTAssertEqualObjects(error.userInfo[NFKRemoteErrorStatusCodeKey], @400);
	XCTAssertTrue([error.localizedDescription containsString:@"no prompt at all"]);
}

@end
