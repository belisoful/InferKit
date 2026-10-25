//
//  NFKServedResponsesTests.m
//  InferKitTests
//
//  OpenAI's Responses API served by NFKInferenceServer, over real HTTP on a loopback port: the input
//  items in the contract's shape, the response object and its event stream, function and custom tool
//  calls both ways, images, errors, keys, token counting, the stored-response routes it does not
//  serve, and NFKRemoteResponsesBackend pointed at the server.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

/*! A synchronous backend whose reply a test writes, recording what reached it. */
@interface NFKResponsesRouteBackend : NSObject <NFKInferenceBackend>
@property (atomic, strong, nullable) NFKInferenceRequest *lastRequest;
@property (nonatomic, copy, nullable) NFKInferenceResult * _Nullable (^reply)(NFKInferenceRequest *request, NSError **error);
@end

@implementation NFKResponsesRouteBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"responses-route-test";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	self.lastRequest = request;
	return self.reply != nil ? self.reply(request, error) : [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"ok" }];
}

@end

/*! A backend that counts a request's tokens as the number of its messages. */
@interface NFKResponsesRouteCountingBackend : NFKResponsesRouteBackend
@end

@implementation NFKResponsesRouteCountingBackend

- (nullable NSNumber *)tokenCountForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	self.lastRequest = request;
	return @(request.messages.count);
}

@end

/*! A backend whose job reports the partials a test writes, then finishes with its result or error. */
@interface NFKResponsesRouteStreamingBackend : NFKResponsesRouteBackend
@property (nonatomic, copy) NSArray<NFKInferenceResult *> *partials;
@property (nonatomic, strong, nullable) NFKInferenceResult *finalResult;
@property (nonatomic, strong, nullable) NSError *finalError;
@end

@implementation NFKResponsesRouteStreamingBackend

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

@interface NFKServedResponsesTests : XCTestCase
@property (nonatomic, assign) NSUInteger savedRetryAttempts;
@end

@implementation NFKServedResponsesTests

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

- (NFKRemoteResponsesBackend *)clientFor:(NFKInferenceServer *)server model:(NSString *)model
{
	NFKRemoteResponsesBackend *client = [NFKRemoteResponsesBackend backendWithEndpointURL:[server.localBaseURL URLByAppendingPathComponent:@"responses"]];
	client.modelName = model;
	return client;
}

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
		request.HTTPBody = [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL];
		[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	}
	for (NSString *name in headers) {
		[request setValue:headers[name] forHTTPHeaderField:name];
	}
	NSHTTPURLResponse *response = nil;
	NSData *data = [NFKRemoteTransport sendRequest:request session:NSURLSession.sharedSession response:&response error:NULL];
	*outStatus = response.statusCode;
	return data != nil ? [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] : nil;
}

- (nullable NSDictionary *)postResponses:(NSDictionary *)body server:(NFKInferenceServer *)server status:(NSInteger *)outStatus
{
	NSString *text = [self send:@"POST" path:@"/v1/responses" body:body headers:nil server:server status:outStatus];
	id reply = text != nil ? [NSJSONSerialization JSONObjectWithData:[text dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL] : nil;
	return [reply isKindOfClass:NSDictionary.class] ? reply : nil;
}

/*! The data of each event in a stream, each event's name checked against its data's type and the
	sequence numbers checked to count up from zero. */
- (NSArray<NSDictionary *> *)eventsIn:(NSString *)stream
{
	NSMutableArray<NSDictionary *> *events = [NSMutableArray array];
	for (NSString *block in [stream componentsSeparatedByString:@"\n\n"]) {
		NSString *name = nil;
		NSDictionary *data = nil;
		for (NSString *line in [block componentsSeparatedByString:@"\n"]) {
			if ([line hasPrefix:@"event: "]) {
				name = [line substringFromIndex:7];
			} else if ([line hasPrefix:@"data: "]) {
				data = [NSJSONSerialization JSONObjectWithData:[[line substringFromIndex:6] dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
			}
		}
		if (name == nil) {
			continue;
		}
		XCTAssertEqualObjects(data[@"type"], name);
		XCTAssertEqualObjects(data[@"sequence_number"], @(events.count));
		[events addObject:data];
	}
	return events;
}

- (NSString *)PNGDataURLWidth:(size_t)width height:(size_t)height
{
	CVPixelBufferRef buffer = NULL;
	NSDictionary *attributes = @{ (id)kCVPixelBufferIOSurfacePropertiesKey: @{} };
	CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef)attributes, &buffer);
	NSString *dataURL = [NFKImageCoding dataURLForImage:(__bridge id)buffer];
	CVPixelBufferRelease(buffer);
	return dataURL;
}

#pragma mark Requests

- (void)testAResponsesBodyReachesTheBackendInTheContractShape
{
	NFKResponsesRouteBackend *backend = [[NFKResponsesRouteBackend alloc] init];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend }];
	NSDictionary *shell = @{ @"type": @"function", @"name": @"shell", @"description": @"Runs a command.", @"strict": @NO,
							 @"parameters": @{ @"type": @"object", @"properties": @{ @"command": @{ @"type": @"array" } } } };
	NSDictionary *patch = @{ @"type": @"custom", @"name": @"apply_patch", @"description": @"Applies a patch.",
							 @"format": @{ @"type": @"grammar", @"syntax": @"lark", @"definition": @"start: /.+/" } };
	NSData *pdf = [@"%PDF-1.4" dataUsingEncoding:NSUTF8StringEncoding];
	NSDictionary *body = @{ @"model": @"served", @"instructions": @"You are a coding agent.", @"store": @NO, @"stream": @NO,
							@"include": @[ @"reasoning.encrypted_content" ], @"parallel_tool_calls": @NO,
							@"tools": @[ shell, patch, @{ @"type": @"web_search" } ],
							@"tool_choice": @{ @"type": @"function", @"name": @"shell" },
							@"temperature": @0.2, @"top_p": @0.8, @"max_output_tokens": @512, @"prompt_cache_key": @"session-1",
							@"prompt_cache_retention": @"24h", @"prompt_cache_options": @{ @"mode": @"explicit", @"ttl": @"12h", @"prewarm": @NO },
							@"reasoning": @{ @"effort": @"high", @"summary": @"auto" }, @"metadata": @{ @"origin": @"test" },
							@"text": @{ @"format": @{ @"type": @"json_schema", @"name": @"reply", @"schema": @{ @"type": @"object" } } },
							@"stopSequences": @[ @"END" ],
							@"input": @[ @{ @"type": @"message", @"role": @"developer", @"content": @[ @{ @"type": @"input_text", @"text": @"Sandbox: read-only." } ] },
										 @{ @"role": @"user", @"content": @[ @{ @"type": @"input_text", @"text": @"Fix the bug." },
																			 @{ @"type": @"input_image", @"image_url": [self PNGDataURLWidth:4 height:2] },
																			 @{ @"type": @"input_file", @"filename": @"spec.pdf",
																				@"file_data": [@"data:application/pdf;base64," stringByAppendingString:[pdf base64EncodedStringWithOptions:0]] } ] },
										 @{ @"type": @"reasoning", @"id": @"rs_1", @"summary": @[], @"encrypted_content": @"opaque" },
										 @{ @"type": @"message", @"role": @"assistant", @"content": @[ @{ @"type": @"output_text", @"text": @"Looking." } ] },
										 @{ @"type": @"function_call", @"call_id": @"call_a", @"name": @"shell", @"arguments": @"{\"command\":[\"ls\"]}" },
										 @{ @"type": @"custom_tool_call", @"call_id": @"call_b", @"name": @"apply_patch", @"input": @"*** Begin Patch" },
										 @{ @"type": @"function_call_output", @"call_id": @"call_a", @"output": @"main.c" },
										 @{ @"type": @"custom_tool_call_output", @"call_id": @"call_b", @"output": @"Done!" },
										 @{ @"role": @"user", @"content": @"Thanks." } ] };
	NSInteger status = 0;
	NSDictionary *reply = [self postResponses:body server:server status:&status];
	XCTAssertEqual(status, 200, @"%@", reply);

	NFKInferenceRequest *received = backend.lastRequest;
	NSArray<NSDictionary *> *messages = received.messages;
	XCTAssertEqual(messages.count, 6u, @"%@", messages);
	XCTAssertEqualObjects(messages[0], (@{ @"role": @"system", @"content": @"You are a coding agent.\n\nSandbox: read-only." }),
						  @"the instructions and the developer turn before the first other turn become one system turn");
	XCTAssertEqualObjects(messages[1], (@{ @"role": @"user", @"content": @"Fix the bug." }));
	XCTAssertEqualObjects(messages[2][@"content"], @"Looking.");
	NSArray<NSDictionary *> *calls = [NFKToolMessages toolCallsOfAssistantMessage:messages[2]];
	XCTAssertEqual(calls.count, 2u, @"the reasoning item is not replayed, and the calls join the assistant turn");
	XCTAssertEqualObjects(calls[0][@"id"], @"call_a");
	XCTAssertEqualObjects(calls[0][@"arguments"], (@{ @"command": @[ @"ls" ] }));
	XCTAssertEqualObjects(calls[1][@"name"], @"apply_patch");
	XCTAssertEqualObjects(calls[1][@"arguments"], (@{ @"input": @"*** Begin Patch" }), @"a custom tool's input is its input argument");
	XCTAssertEqualObjects(messages[3][@"tool_call_id"], @"call_a");
	XCTAssertEqualObjects([NFKToolMessages textOfToolMessage:messages[3]], @"main.c");
	XCTAssertEqualObjects([NFKToolMessages toolNameOfMessage:messages[3] inConversation:messages], @"shell");
	XCTAssertEqualObjects([NFKToolMessages toolNameOfMessage:messages[4] inConversation:messages], @"apply_patch");
	XCTAssertEqualObjects(messages[5], (@{ @"role": @"user", @"content": @"Thanks." }));

	CVPixelBufferRef image = (__bridge CVPixelBufferRef)[received inputForKey:NFKInputImage];
	XCTAssertTrue(image != NULL && CFGetTypeID(image) == CVPixelBufferGetTypeID());
	XCTAssertEqual(CVPixelBufferGetWidth(image), 4u);
	XCTAssertEqualObjects([received inputForKey:NFKInputDocument], pdf);

	NSArray *tools = [received parameterForKey:NFKParameterTools];
	XCTAssertEqual(tools.count, 2u, @"the built-in web search is not offered: %@", tools);
	XCTAssertEqualObjects(tools[0], (@{ @"name": @"shell", @"description": @"Runs a command.", @"parameters": shell[@"parameters"] }));
	XCTAssertEqualObjects(tools[1][@"name"], @"apply_patch");
	XCTAssertEqualObjects(tools[1][@"parameters"][@"required"], (@[ @"input" ]));
	XCTAssertEqualObjects([received parameterForKey:@"tool_choice"], (@{ @"type": @"function", @"function": @{ @"name": @"shell" } }));
	XCTAssertEqualObjects([received parameterForKey:NFKParameterTemperature], @0.2);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterTopP], @0.8);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterMaxTokens], @512);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterConversationKey], @"session-1");
	XCTAssertEqualObjects([received parameterForKey:NFKParameterPromptCacheRetention], @"24h");
	XCTAssertEqualObjects([received parameterForKey:NFKParameterPromptCacheMode], @"explicit");
	XCTAssertEqualObjects([received parameterForKey:NFKParameterPromptCacheTTL], @"12h");
	XCTAssertEqualObjects([received parameterForKey:NFKParameterPromptCachePrewarm], @NO);
	XCTAssertNil([received parameterForKey:@"prompt_cache_options"]);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterReasoningEffort], NFKReasoningEffortDeep);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterJSONSchema], (@{ @"type": @"object" }));
	XCTAssertEqualObjects([received parameterForKey:NFKParameterStopSequences], (@[ @"END" ]), @"a contract key the client sends under its own name passes through");
	XCTAssertEqualObjects([received parameterForKey:@"parallel_tool_calls"], @NO);
	for (NSString *consumed in @[ @"store", @"include", @"metadata", @"instructions", @"input", @"reasoning", @"text" ]) {
		XCTAssertNil([received parameterForKey:consumed], @"%@ is read, not passed on", consumed);
	}
}

#pragma mark Replies

- (void)testAReplyIsAResponseObjectThatIsNotStored
{
	NFKResponsesRouteBackend *backend = [[NFKResponsesRouteBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"Hello.",
														NFKOutputCitations: @[ @{ @"url": @"https://example.com", @"title": @"Example", @"start": @0, @"end": @5 } ],
														NFKOutputUsage: @{ NFKUsageInputTokens: @20, NFKUsageCachedTokens: @8,
																		   NFKUsageOutputTokens: @4, NFKUsageReasoningTokens: @1 } }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend }];
	NSInteger status = 0;
	NSDictionary *reply = [self postResponses:@{ @"model": @"served", @"input": @"Hi", @"instructions": @"Be kind.", @"max_output_tokens": @50 }
									   server:server status:&status];
	XCTAssertEqual(status, 200, @"%@", reply);
	XCTAssertTrue([reply[@"id"] hasPrefix:@"resp_"]);
	XCTAssertEqualObjects(reply[@"object"], @"response");
	XCTAssertEqualObjects(reply[@"status"], @"completed");
	XCTAssertEqualObjects(reply[@"model"], @"served");
	XCTAssertEqualObjects(reply[@"store"], @NO, @"the server keeps no responses");
	XCTAssertEqualObjects(reply[@"previous_response_id"], NSNull.null);
	XCTAssertEqualObjects(reply[@"instructions"], @"Be kind.");
	XCTAssertEqualObjects(reply[@"max_output_tokens"], @50);
	XCTAssertEqualObjects(reply[@"error"], NSNull.null);
	XCTAssertEqualObjects(reply[@"incomplete_details"], NSNull.null);
	NSArray *output = reply[@"output"];
	XCTAssertEqual(output.count, 1u, @"%@", output);
	XCTAssertEqualObjects(output[0][@"type"], @"message");
	XCTAssertEqualObjects(output[0][@"role"], @"assistant");
	XCTAssertEqualObjects(output[0][@"status"], @"completed");
	XCTAssertTrue([output[0][@"id"] hasPrefix:@"msg_"]);
	NSDictionary *part = [output[0][@"content"] firstObject];
	XCTAssertEqualObjects(part[@"type"], @"output_text");
	XCTAssertEqualObjects(part[@"text"], @"Hello.");
	XCTAssertEqualObjects(part[@"annotations"], (@[ @{ @"type": @"url_citation", @"url": @"https://example.com", @"title": @"Example",
													  @"start_index": @0, @"end_index": @5 } ]));
	XCTAssertEqualObjects(reply[@"usage"], (@{ @"input_tokens": @20, @"input_tokens_details": @{ @"cached_tokens": @8 },
											   @"output_tokens": @4, @"output_tokens_details": @{ @"reasoning_tokens": @1 },
											   @"total_tokens": @24 }));
	XCTAssertEqualObjects(backend.lastRequest.messages, (@[ @{ @"role": @"system", @"content": @"Be kind." }, @{ @"role": @"user", @"content": @"Hi" } ]));
}

- (void)testCallsComeBackAsFunctionAndCustomToolCallItems
{
	NFKResponsesRouteBackend *backend = [[NFKResponsesRouteBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputReasoning: @"Need to look.",
														NFKOutputToolCalls: @[ @{ @"id": @"call_1", @"name": @"shell", @"arguments": @{ @"command": @[ @"ls" ] } },
																			   @{ @"id": @"call_2", @"name": @"apply_patch", @"arguments": @{ @"input": @"*** Begin Patch" } } ] }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend }];
	NSDictionary *body = @{ @"model": @"served", @"input": @"List and patch.",
							@"tools": @[ @{ @"type": @"function", @"name": @"shell", @"parameters": @{ @"type": @"object" } },
										 @{ @"type": @"custom", @"name": @"apply_patch" } ] };
	NSInteger status = 0;
	NSDictionary *reply = [self postResponses:body server:server status:&status];
	NSArray *output = reply[@"output"];
	XCTAssertEqual(output.count, 3u, @"no message item when the reply has no text: %@", output);
	XCTAssertEqualObjects(output[0][@"type"], @"reasoning");
	XCTAssertEqualObjects(output[0][@"summary"], (@[ @{ @"type": @"summary_text", @"text": @"Need to look." } ]));
	XCTAssertEqualObjects(output[1][@"type"], @"function_call");
	XCTAssertEqualObjects(output[1][@"call_id"], @"call_1");
	XCTAssertEqualObjects(output[1][@"name"], @"shell");
	XCTAssertEqualObjects(output[1][@"arguments"], @"{\"command\":[\"ls\"]}");
	XCTAssertEqualObjects(output[1][@"status"], @"completed");
	XCTAssertEqualObjects(output[2][@"type"], @"custom_tool_call");
	XCTAssertEqualObjects(output[2][@"call_id"], @"call_2");
	XCTAssertEqualObjects(output[2][@"input"], @"*** Begin Patch");
	XCTAssertEqualObjects(reply[@"usage"], NSNull.null, @"a backend that reports no usage leaves it null");
}

- (void)testReachingMaxOutputTokensIsAnIncompleteResponse
{
	NFKResponsesRouteBackend *backend = [[NFKResponsesRouteBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"Once", NFKOutputUsage: @{ NFKUsageInputTokens: @3, NFKUsageOutputTokens: @16 } }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend }];
	NSInteger status = 0;
	NSDictionary *reply = [self postResponses:@{ @"model": @"served", @"input": @"A story.", @"max_output_tokens": @16 } server:server status:&status];
	XCTAssertEqualObjects(reply[@"status"], @"incomplete");
	XCTAssertEqualObjects(reply[@"incomplete_details"], (@{ @"reason": @"max_output_tokens" }));
}

#pragma mark Streaming

- (void)testAStreamSendsTheResponsesEventsInOrder
{
	NFKResponsesRouteStreamingBackend *backend = [[NFKResponsesRouteStreamingBackend alloc] init];
	backend.partials = @[ [NFKInferenceResult resultWithOutputs:@{ NFKOutputReasoning: @"Think" }],
						  [NFKInferenceResult resultWithOutputs:@{ NFKOutputReasoning: @"Think more", NFKOutputText: @"Run" }],
						  [NFKInferenceResult resultWithOutputs:@{ NFKOutputReasoning: @"Think more", NFKOutputText: @"Running ls." }] ];
	backend.finalResult = [NFKInferenceResult resultWithOutputs:@{ NFKOutputReasoning: @"Think more", NFKOutputText: @"Running ls.",
																   NFKOutputToolCalls: @[ @{ @"id": @"call_f", @"name": @"shell", @"arguments": @{ @"command": @[ @"ls" ] } },
																						  @{ @"id": @"call_c", @"name": @"apply_patch", @"arguments": @{ @"input": @"patch" } } ],
																   NFKOutputUsage: @{ NFKUsageInputTokens: @7, NFKUsageOutputTokens: @5 } }];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"streamer": backend }];
	NSDictionary *body = @{ @"model": @"streamer", @"input": @"List files.", @"stream": @YES,
							@"tools": @[ @{ @"type": @"function", @"name": @"shell" }, @{ @"type": @"custom", @"name": @"apply_patch" } ] };
	NSInteger status = 0;
	NSString *stream = [self send:@"POST" path:@"/v1/responses" body:body headers:nil server:server status:&status];
	XCTAssertEqual(status, 200);
	NSArray<NSDictionary *> *events = [self eventsIn:stream];
	NSArray<NSString *> *expected = @[ @"response.created", @"response.in_progress",
									   @"response.output_item.added", @"response.reasoning_summary_part.added",
									   @"response.reasoning_summary_text.delta", @"response.reasoning_summary_text.delta",
									   @"response.reasoning_summary_text.done", @"response.reasoning_summary_part.done", @"response.output_item.done",
									   @"response.output_item.added", @"response.content_part.added",
									   @"response.output_text.delta", @"response.output_text.delta",
									   @"response.output_text.done", @"response.content_part.done", @"response.output_item.done",
									   @"response.output_item.added", @"response.function_call_arguments.delta",
									   @"response.function_call_arguments.done", @"response.output_item.done",
									   @"response.output_item.added", @"response.custom_tool_call_input.delta",
									   @"response.custom_tool_call_input.done", @"response.output_item.done",
									   @"response.completed" ];
	XCTAssertEqualObjects([events valueForKey:@"type"], expected, @"%@", stream);
	if (events.count != expected.count) {
		return;
	}
	XCTAssertEqualObjects(events[0][@"response"][@"status"], @"in_progress");
	XCTAssertEqualObjects(events[4][@"delta"], @"Think");
	XCTAssertEqualObjects(events[5][@"delta"], @" more");
	XCTAssertEqualObjects(events[8][@"item"][@"summary"], (@[ @{ @"type": @"summary_text", @"text": @"Think more" } ]));
	XCTAssertEqualObjects(events[9][@"output_index"], @1);
	XCTAssertEqualObjects(events[11][@"delta"], @"Run");
	XCTAssertEqualObjects(events[12][@"delta"], @"ning ls.");
	XCTAssertEqualObjects(events[11][@"item_id"], events[9][@"item"][@"id"], @"a delta names its item");
	XCTAssertEqualObjects(events[13][@"text"], @"Running ls.");
	XCTAssertEqualObjects(events[16][@"item"][@"arguments"], @"", @"a call is announced with empty arguments");
	XCTAssertEqualObjects(events[17][@"delta"], @"{\"command\":[\"ls\"]}");
	XCTAssertEqualObjects(events[19][@"item"][@"call_id"], @"call_f");
	XCTAssertEqualObjects(events[23][@"item"][@"input"], @"patch");
	NSDictionary *completed = events[24][@"response"];
	XCTAssertEqualObjects(completed[@"status"], @"completed");
	XCTAssertEqualObjects([completed[@"output"] valueForKey:@"type"], (@[ @"reasoning", @"message", @"function_call", @"custom_tool_call" ]));
	XCTAssertEqualObjects([completed[@"output"] valueForKey:@"id"],
						  (@[ events[8][@"item"][@"id"], events[15][@"item"][@"id"], events[19][@"item"][@"id"], events[23][@"item"][@"id"] ]),
						  @"the completed response lists the items the stream finished");
	XCTAssertEqualObjects(completed[@"usage"][@"output_tokens"], @5);
}

- (void)testAFailureAfterTheStreamOpensIsAFailedResponse
{
	NFKResponsesRouteStreamingBackend *backend = [[NFKResponsesRouteStreamingBackend alloc] init];
	backend.partials = @[ [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"partial" }] ];
	backend.finalError = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceRateLimited
										 userInfo:@{ NSLocalizedDescriptionKey: @"out of quota" }];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"flaky": backend }];
	NSInteger status = 0;
	NSString *stream = [self send:@"POST" path:@"/v1/responses" body:@{ @"model": @"flaky", @"input": @"Go", @"stream": @YES }
						  headers:nil server:server status:&status];
	XCTAssertEqual(status, 200);
	NSDictionary *last = [self eventsIn:stream].lastObject;
	XCTAssertEqualObjects(last[@"type"], @"response.failed", @"%@", stream);
	XCTAssertEqualObjects(last[@"response"][@"status"], @"failed");
	XCTAssertEqualObjects(last[@"response"][@"error"][@"code"], @"rate_limit_exceeded");
	XCTAssertEqualObjects(last[@"response"][@"error"][@"message"], @"out of quota");

	backend.partials = @[];
	stream = [self send:@"POST" path:@"/v1/responses" body:@{ @"model": @"flaky", @"input": @"Go", @"stream": @YES }
				headers:nil server:server status:&status];
	XCTAssertEqual(status, 429, @"a failure before the first event is the reply's status");
	XCTAssertTrue([stream containsString:@"\"rate_limit_error\""], @"%@", stream);
}

#pragma mark Errors and keys

- (void)testWhatTheServerDoesNotKeepIsRefused
{
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": [[NFKResponsesRouteBackend alloc] init],
															   @"other": [[NFKResponsesRouteBackend alloc] init] }];
	NSInteger status = 0;
	NSDictionary *reply = [self postResponses:@{ @"model": @"served", @"input": @"Hi", @"previous_response_id": @"resp_old" } server:server status:&status];
	XCTAssertEqual(status, 400);
	XCTAssertTrue([reply[@"error"][@"message"] containsString:@"keeps no responses"], @"%@", reply);

	reply = [self postResponses:@{ @"model": @"served", @"input": @[ @{ @"type": @"item_reference", @"id": @"msg_old" } ] } server:server status:&status];
	XCTAssertEqual(status, 400);

	NSString *text = [self send:@"GET" path:@"/v1/responses/resp_old" body:nil headers:nil server:server status:&status];
	XCTAssertEqual(status, 404);
	XCTAssertTrue([text containsString:@"keeps no responses"], @"%@", text);
	[self send:@"DELETE" path:@"/v1/responses/resp_old" body:nil headers:nil server:server status:&status];
	XCTAssertEqual(status, 404);

	reply = [self postResponses:@{ @"model": @"absent", @"input": @"Hi" } server:server status:&status];
	XCTAssertEqual(status, 404);
	XCTAssertEqualObjects(reply[@"error"][@"type"], @"server_error");

	reply = [self postResponses:@{ @"model": @"served", @"instructions": @"Only this." } server:server status:&status];
	XCTAssertEqual(status, 400, @"instructions alone are no input");
	XCTAssertEqualObjects(reply[@"error"][@"type"], @"invalid_request_error");

	reply = [self postResponses:@{ @"model": @"served", @"input": @[ @{ @"type": @"function_call_output", @"call_id": @"nowhere", @"output": @"x" } ] }
						 server:server status:&status];
	XCTAssertEqual(status, 400, @"an output answers a call in the conversation");

	reply = [self postResponses:@{ @"model": @"served", @"input": @[ @{ @"role": @"user", @"content": @[ @{ @"type": @"input_image", @"file_id": @"file-1" } ] } ] }
						 server:server status:&status];
	XCTAssertEqual(status, 400);
	XCTAssertTrue([reply[@"error"][@"message"] containsString:@"keeps no files"], @"%@", reply);
}

- (void)testTheResponsesRouteTakesABearerKey
{
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": [[NFKResponsesRouteBackend alloc] init] }
											  configuration:^(NFKInferenceServer *configured) {
		configured.loopbackExemptScopes = 0;
		[configured addAPIKey:@"key-inference" named:@"codex" scopes:NFKServerKeyScopeInference];
	}];
	NSInteger status = 0;
	[self postResponses:@{ @"model": @"served", @"input": @"Hi" } server:server status:&status];
	XCTAssertEqual(status, 401);
	[self send:@"POST" path:@"/v1/responses" body:@{ @"model": @"served", @"input": @"Hi" } headers:@{ @"Authorization": @"Bearer key-inference" }
		server:server status:&status];
	XCTAssertEqual(status, 200);
}

#pragma mark Token counting

- (void)testInputTokensAnswersFromABackendThatCounts
{
	NFKInferenceServer *server = [self startedServerHosting:@{ @"counts": [[NFKResponsesRouteCountingBackend alloc] init],
															   @"silent": [[NFKResponsesRouteBackend alloc] init] }];
	NSInteger status = 0;
	NSString *text = [self send:@"POST" path:@"/v1/responses/input_tokens"
						   body:@{ @"model": @"counts", @"instructions": @"Be brief.", @"input": @"Hello" } headers:nil server:server status:&status];
	XCTAssertEqual(status, 200, @"%@", text);
	NSDictionary *reply = [NSJSONSerialization JSONObjectWithData:[text dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
	XCTAssertEqualObjects(reply, (@{ @"object": @"response.input_tokens", @"input_tokens": @2 }));
	[self send:@"POST" path:@"/v1/responses/input_tokens" body:@{ @"model": @"silent", @"input": @"Hello" } headers:nil server:server status:&status];
	XCTAssertEqual(status, 400);
}

#pragma mark NFKRemoteResponsesBackend against the server

- (void)testTheResponsesClientRunsAHostedModel
{
	NFKResponsesRouteBackend *backend = [[NFKResponsesRouteBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"Bonjour.", NFKOutputReasoning: @"French greeting.",
														NFKOutputUsage: @{ NFKUsageInputTokens: @12, NFKUsageCachedTokens: @4, NFKUsageOutputTokens: @2 } }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend }];
	NFKRemoteResponsesBackend *client = [self clientFor:server model:@"served"];
	NSArray *messages = @[ @{ @"role": @"system", @"content": @"Answer in French." }, @{ @"role": @"user", @"content": @"Hello" } ];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: messages }
															   parameters:@{ NFKParameterMaxTokens: @40, NFKParameterTemperature: @0.4,
																			 NFKParameterTopP: @0.7, NFKParameterTopK: @20,
																			 NFKParameterReasoningEffort: NFKReasoningEffortModerate }
														   outputModality:NFKModalityText];
	NSError *error = nil;
	NFKInferenceResult *result = [client runInferenceForRequest:request error:&error];
	XCTAssertEqualObjects(result.text, @"Bonjour.", @"%@", error);
	XCTAssertEqualObjects([result outputForKey:NFKOutputReasoning], @"French greeting.");
	XCTAssertTrue([[result outputForKey:NFKOutputResponseIdentifier] hasPrefix:@"resp_"]);
	XCTAssertEqualObjects([result outputForKey:NFKOutputUsage], (@{ NFKUsageInputTokens: @12, NFKUsageCachedTokens: @4, NFKUsageOutputTokens: @2 }));
	NFKInferenceRequest *received = backend.lastRequest;
	XCTAssertEqualObjects(received.messages, messages);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterMaxTokens], @40);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterTemperature], @0.4);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterTopP], @0.7);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterTopK], @20);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterReasoningEffort], NFKReasoningEffortModerate);

	client.runsInBackground = YES;
	XCTAssertEqualObjects([client runInferenceForRequest:request error:&error].text, @"Bonjour.",
						  @"a background request is answered at once, completed, so the client does not poll: %@", error);

	NFKInferenceRequest *chained = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"And then?" }
																parameters:@{ NFKParameterPreviousResponseIdentifier: @"resp_old" }
															outputModality:NFKModalityText];
	client.runsInBackground = NO;
	XCTAssertNil([client runInferenceForRequest:chained error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported, @"%@", error);
}

- (void)testTheResponsesClientStreamsTheHostedJob
{
	NFKResponsesRouteStreamingBackend *backend = [[NFKResponsesRouteStreamingBackend alloc] init];
	backend.partials = @[ [NFKInferenceResult resultWithOutputs:@{ NFKOutputReasoning: @"Hm" }],
						  [NFKInferenceResult resultWithOutputs:@{ NFKOutputReasoning: @"Hm", NFKOutputText: @"Hel" }],
						  [NFKInferenceResult resultWithOutputs:@{ NFKOutputReasoning: @"Hm", NFKOutputText: @"Hello" }] ];
	backend.finalResult = [NFKInferenceResult resultWithOutputs:@{ NFKOutputReasoning: @"Hm", NFKOutputText: @"Hello, world",
																   NFKOutputUsage: @{ NFKUsageInputTokens: @3, NFKUsageOutputTokens: @3 } }];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"streamer": backend }];
	NFKRemoteResponsesBackend *client = [self clientFor:server model:@"streamer"];
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
	XCTAssertEqualObjects([job.result outputForKey:NFKOutputReasoning], @"Hm");
	XCTAssertEqualObjects([job.result outputForKey:NFKOutputUsage][NFKUsageOutputTokens], @3);
	XCTAssertGreaterThan(seen.count, 1u, @"the text arrived in more than one piece");
	XCTAssertEqualObjects(seen.lastObject, @"Hello, world");
}

- (void)testTheResponsesClientsToolConversationRoundTrips
{
	NFKResponsesRouteBackend *backend = [[NFKResponsesRouteBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		if ([request.messages.lastObject[@"role"] isEqual:@"tool"]) {
			return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"It is 18 degrees in Lyon." }];
		}
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputReasoning: @"Look it up.", NFKOutputText: @"Checking.",
														NFKOutputToolCalls: @[ @{ @"id": @"call_lyon", @"name": @"get_weather",
																				  @"arguments": @{ @"city": @"Lyon" } } ] }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend }];
	NFKRemoteResponsesBackend *client = [self clientFor:server model:@"served"];
	NSDictionary *tool = @{ @"name": @"get_weather", @"description": @"The weather in a city.",
							@"parameters": @{ @"type": @"object", @"properties": @{ @"city": @{ @"type": @"string" } } } };
	NSArray *opening = @[ @{ @"role": @"user", @"content": @"Weather in Lyon?" } ];
	NSError *error = nil;
	NFKInferenceResult *first = [client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: opening }
																					   parameters:@{ NFKParameterTools: @[ tool ] }
																				   outputModality:NFKModalityText] error:&error];
	XCTAssertEqualObjects(first.toolCalls.firstObject[@"id"], @"call_lyon", @"%@", error);
	XCTAssertEqualObjects(first.toolCalls.firstObject[@"arguments"], (@{ @"city": @"Lyon" }));
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
	XCTAssertEqual(received.count, 3u, @"the replayed output items rebuild one assistant turn: %@", received);
	XCTAssertEqualObjects(received[1][@"content"], @"Checking.");
	XCTAssertEqualObjects([NFKToolMessages toolCallsOfAssistantMessage:received[1]].firstObject[@"id"], @"call_lyon");
	XCTAssertEqualObjects([NFKToolMessages textOfToolMessage:received[2]], @"18 C");
	XCTAssertEqualObjects([NFKToolMessages toolNameOfMessage:received[2] inConversation:received], @"get_weather");
}

- (void)testTheResponsesClientStreamsAToolCall
{
	NFKResponsesRouteStreamingBackend *backend = [[NFKResponsesRouteStreamingBackend alloc] init];
	backend.partials = @[];
	backend.finalResult = [NFKInferenceResult resultWithOutputs:@{ NFKOutputToolCalls: @[ @{ @"id": @"call_s", @"name": @"lookup",
																							 @"arguments": @{ @"term": @"orbit" } } ] }];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"streamer": backend }];
	NFKRemoteResponsesBackend *client = [self clientFor:server model:@"streamer"];
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

- (void)testTheResponsesClientsImageAndSchemaArrive
{
	NFKResponsesRouteBackend *backend = [[NFKResponsesRouteBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"{\"color\":\"grey\"}" }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"vision": backend }];
	NFKRemoteResponsesBackend *client = [self clientFor:server model:@"vision"];
	CVPixelBufferRef image = NULL;
	NSDictionary *attributes = @{ (id)kCVPixelBufferIOSurfacePropertiesKey: @{} };
	CVPixelBufferCreate(kCFAllocatorDefault, 6, 3, kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef)attributes, &image);
	NSDictionary *schema = @{ @"type": @"object", @"properties": @{ @"color": @{ @"type": @"string" } } };
	NSError *error = nil;
	NFKInferenceResult *result = [client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Its color?",
																										 NFKInputImage: (__bridge id)image }
																						parameters:@{ NFKParameterJSONSchema: schema }
																					outputModality:NFKModalityText] error:&error];
	CVPixelBufferRelease(image);
	XCTAssertEqualObjects(result.structured, (@{ @"color": @"grey" }), @"%@", error);
	CVPixelBufferRef received = (__bridge CVPixelBufferRef)[backend.lastRequest inputForKey:NFKInputImage];
	XCTAssertTrue(received != NULL && CFGetTypeID(received) == CVPixelBufferGetTypeID());
	XCTAssertEqual(CVPixelBufferGetWidth(received), 6u);
	XCTAssertEqual(CVPixelBufferGetHeight(received), 3u);
	XCTAssertEqualObjects([backend.lastRequest parameterForKey:NFKParameterJSONSchema], schema);
	XCTAssertEqualObjects(backend.lastRequest.prompt, @"Its color?");
}

- (void)testTheResponsesClientReadsTheServedErrorCode
{
	NFKResponsesRouteBackend *backend = [[NFKResponsesRouteBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceRateLimited
								 userInfo:@{ NSLocalizedDescriptionKey: @"slow down" }];
		return nil;
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"failing": backend }];
	NFKRemoteResponsesBackend *client = [self clientFor:server model:@"failing"];
	NSError *error = nil;
	XCTAssertNil([client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceRateLimited, @"%@", error);
	XCTAssertEqualObjects(error.userInfo[NFKRemoteErrorStatusCodeKey], @429);
	XCTAssertTrue([error.localizedDescription containsString:@"slow down"]);
}

@end
