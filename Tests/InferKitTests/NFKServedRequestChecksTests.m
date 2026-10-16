//
//  NFKServedRequestChecksTests.m
//  InferKitTests
//
//  What the server checks of a request before it runs, over real HTTP on a loopback port: every
//  documented body field of every JSON route refused with 400 when it carries another type, the
//  contract types of what reaches a backend, form fields, a browser Origin and allowedOrigins, a body
//  read as JSON whatever its Content-Type, an exception answered with 500 while the server keeps
//  serving, and MCP model-tool calls that hold no thread while they wait.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
#import <arpa/inet.h>
#import <sys/socket.h>
#import <unistd.h>

/*! A synchronous backend that answers ok, after a pause a test sets, or raises when a test asks. */
@interface NFKRequestChecksBackend : NSObject <NFKInferenceBackend>
@property (atomic, assign) NSTimeInterval pause;
@property (atomic, assign) BOOL raises;
@property (atomic, strong, nullable) NFKInferenceResult *result;
@property (atomic, assign) NSUInteger runs;
@property (atomic, strong, nullable) NFKInferenceRequest *lastRequest;
@end

@implementation NFKRequestChecksBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"request-checks-test";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	@synchronized (self) {
		self.runs += 1;
	}
	self.lastRequest = request;
	if (self.pause > 0) {
		[NSThread sleepForTimeInterval:self.pause];
	}
	if (self.raises) {
		[NSException raise:NSInvalidArgumentException format:@"the test backend raises"];
	}
	return self.result ?: [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"ok", NFKOutputEmbedding: @[ @0.5, @0.25 ] }];
}

- (nullable NSNumber *)tokenCountForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	return @(request.messages.count);
}

@end

@interface NFKServedRequestChecksTests : XCTestCase
@property (nonatomic, strong) NFKInferenceServer *server;
@property (nonatomic, strong) NFKRequestChecksBackend *backend;
@end

@implementation NFKServedRequestChecksTests

- (void)setUp
{
	[super setUp];
	self.backend = [[NFKRequestChecksBackend alloc] init];
	self.server = [[NFKInferenceServer alloc] init];
	self.server.port = 0;
	self.server.loopbackOnly = YES;
	self.server.advertisesService = NO;
	[self.server addBackend:self.backend forModelName:@"served"];
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

/*! Sends data to path with headers, and returns the parsed reply and its status. */
- (nullable id)send:(NSString *)method path:(NSString *)path data:(nullable NSData *)data
			headers:(nullable NSDictionary<NSString *, NSString *> *)headers status:(NSInteger *)outStatus
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:[self URLForPath:path]];
	request.HTTPMethod = method;
	request.HTTPBody = data;
	request.timeoutInterval = 30;
	[headers enumerateKeysAndObjectsUsingBlock:^(NSString *field, NSString *value, BOOL *stop) {
		[request setValue:value forHTTPHeaderField:field];
	}];
	NSHTTPURLResponse *response = nil;
	NSData *reply = [NFKRemoteTransport sendRequest:request session:NSURLSession.sharedSession response:&response error:NULL];
	*outStatus = response.statusCode;
	return reply.length > 0 ? [NSJSONSerialization JSONObjectWithData:reply options:0 error:NULL] : nil;
}

/*! Posts body as JSON to path and returns the parsed reply and its status. */
- (nullable id)post:(id)body path:(NSString *)path status:(NSInteger *)outStatus
{
	NSData *data = [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL];
	return [self send:@"POST" path:path data:data headers:@{ @"Content-Type": @"application/json" } status:outStatus];
}

- (NSDictionary *)chatBody
{
	return @{ @"model": @"served", @"messages": @[ @{ @"role": @"user", @"content": @"hi" } ] };
}

/*! Asserts that the server still answers an ordinary chat request. */
- (void)assertStillServes:(NSString *)context
{
	NSInteger status = 0;
	NSDictionary *reply = [self post:[self chatBody] path:@"/v1/chat/completions" status:&status];
	XCTAssertEqual(status, 200, @"the server serves the next request after %@", context);
	XCTAssertEqualObjects(reply[@"choices"][0][@"message"][@"content"], @"ok");
}

/*! The non-null JSON values that are not of type, a type as the server's field lists write it: one or
	more of array, string, object, boolean, number, and integer joined by |. Null is an absent field. */
- (NSArray *)valuesNotOfType:(NSString *)type
{
	NSArray<NSString *> *types = [type componentsSeparatedByString:@"|"];
	BOOL numeric = [types containsObject:@"number"] || [types containsObject:@"integer"];
	NSArray<NSArray *> *candidates = @[ @[ @([types containsObject:@"array"]), @[ @1 ] ],
										@[ @([types containsObject:@"string"]), @"text" ],
										@[ @([types containsObject:@"object"]), @{ @"key": @1 } ],
										@[ @([types containsObject:@"boolean"]), @YES ],
										@[ @([types containsObject:@"number"]), @1.5 ],
										@[ @(numeric), @7 ] ];
	NSMutableArray *values = [NSMutableArray array];
	for (NSArray *candidate in candidates) {
		if (![candidate[0] boolValue]) {
			[values addObject:candidate[1]];
		}
	}
	return values;
}

/*! The fields a route needs, which a null leaves missing. */
- (NSSet<NSString *> *)requiredFieldsOfPath:(NSString *)path
{
	NSDictionary<NSString *, NSString *> *required = @{ @"/v1/chat/completions": @"messages", @"/v1/embeddings": @"input",
														 @"/v1/audio/speech": @"input", @"/v1/images/generations": @"prompt",
														 @"/v1/messages": @"messages", @"/v1/messages/count_tokens": @"messages",
														 @"/v1/responses": @"input", @"/v1/responses/input_tokens": @"input",
														 @"/v1/inferkit/run": @"request" };
	return required[path] != nil ? [NSSet setWithObject:required[path]] : [NSSet set];
}

/*! Posts base with each documented field set to each non-null value not of its type, asserting a 400
	that names the field; then with each field null, asserting that an optional one reads as absent and
	a required one as missing; then that the server still serves. */
- (void)fuzzPath:(NSString *)path base:(NSDictionary *)base fields:(NSDictionary<NSString *, NSString *> *)fields anthropic:(BOOL)anthropic
{
	NSSet<NSString *> *required = [self requiredFieldsOfPath:path];
	for (NSString *field in fields) {
		NSMutableDictionary *nulled = [base mutableCopy];
		nulled[field] = NSNull.null;
		NSInteger nullStatus = 0;
		NSDictionary *nullReply = [self post:nulled path:path status:&nullStatus];
		if ([required containsObject:field]) {
			XCTAssertEqual(nullStatus, 400, @"%@: a null %@ is missing", path, field);
		} else {
			XCTAssertNotEqual(nullStatus, 400, @"%@: a null %@ is absent: %@", path, field, nullReply);
		}
		XCTAssertNotEqual(nullStatus, 0, @"%@ answers a null %@", path, field);
		for (id value in [self valuesNotOfType:fields[field]]) {
			NSMutableDictionary *body = [base mutableCopy];
			body[field] = value;
			NSInteger status = 0;
			NSDictionary *reply = [self post:body path:path status:&status];
			XCTAssertEqual(status, 400, @"%@ %@ = %@", path, field, value);
			NSString *message = reply[@"error"][@"message"];
			XCTAssertTrue([message containsString:field], @"%@ %@: %@", path, field, message);
			if (anthropic) {
				XCTAssertEqualObjects(reply[@"type"], @"error", @"%@ answers in Anthropic's shape", path);
				XCTAssertEqualObjects(reply[@"error"][@"type"], @"invalid_request_error");
			}
		}
	}
	[self assertStillServes:path];
}

#pragma mark Flags

- (void)testEveryStreamFlagOfAnotherTypeIsRefusedAndANullOneIsAbsent
{
	NSArray *wrong = @[ @[ @YES ], @"true", @{ @"on": @YES } ];
	NSDictionary *messages = @{ @"model": @"served", @"max_tokens": @16, @"messages": @[ @{ @"role": @"user", @"content": @"hi" } ] };
	NSDictionary<NSString *, NSDictionary *> *bases = @{ @"/v1/chat/completions": [self chatBody],
														 @"/v1/messages": messages,
														 @"/v1/responses": @{ @"model": @"served", @"input": @"hi" },
														 @"/v1/images/generations": @{ @"model": @"served", @"prompt": @"a cat" },
														 @"/v1/audio/speech": @{ @"model": @"served", @"input": @"hi" },
														 @"/v1/inferkit/run": @{ @"model": @"served", @"request": @{ @"inputs": @{ @"prompt": @"hi" } } } };
	for (NSString *path in bases) {
		for (id value in wrong) {
			NSMutableDictionary *body = [bases[path] mutableCopy];
			body[@"stream"] = value;
			NSInteger status = 0;
			NSDictionary *reply = [self post:body path:path status:&status];
			XCTAssertEqual(status, 400, @"%@ stream = %@", path, value);
			XCTAssertTrue([reply[@"error"][@"message"] containsString:@"stream"], @"%@", reply);
		}
	}
	XCTAssertEqual(self.backend.runs, 0u, @"no refused request reached the backend");
	for (NSString *path in bases) {
		NSMutableDictionary *body = [bases[path] mutableCopy];
		body[@"stream"] = NSNull.null;
		NSInteger status = 0;
		id reply = [self post:body path:path status:&status];
		XCTAssertNotEqual(status, 400, @"%@: a null stream is absent", path);
		XCTAssertTrue([reply isKindOfClass:NSDictionary.class], @"%@ answers in one piece, unstreamed: %@", path, reply);
		if (![path hasPrefix:@"/v1/images"] && ![path hasPrefix:@"/v1/audio"]) {
			XCTAssertEqual(status, 200, @"%@: %@", path, reply);
		}
	}
	[self assertStillServes:@"the refused flags"];
}

- (void)testNestedFlagsOfAnotherTypeAreRefusedAndNullOnesAreAbsent
{
	NSDictionary *call = @{ @"type": @"tool_use", @"id": @"toolu_1", @"name": @"get_weather", @"input": @{} };
	for (id value in @[ NSNull.null, @[ @YES ], @"true", @{ @"on": @YES } ]) {
		BOOL absent = value == NSNull.null;
		NSInteger expected = absent ? 200 : 400;
		NSInteger status = 0;
		NSDictionary *result = @{ @"type": @"tool_result", @"tool_use_id": @"toolu_1", @"content": @"18 C", @"is_error": value };
		NSDictionary *body = @{ @"model": @"served", @"max_tokens": @16,
								@"messages": @[ @{ @"role": @"user", @"content": @"weather?" }, @{ @"role": @"assistant", @"content": @[ call ] },
												@{ @"role": @"user", @"content": @[ result ] } ] };
		NSDictionary *reply = [self post:body path:@"/v1/messages" status:&status];
		XCTAssertEqual(status, expected, @"is_error = %@: %@", value, reply);
		XCTAssertTrue(absent || [reply[@"error"][@"message"] containsString:@"is_error"], @"%@", reply);

		NSMutableDictionary *choice = [@{ @"model": @"served", @"max_tokens": @16, @"messages": @[ @{ @"role": @"user", @"content": @"hi" } ],
										  @"tool_choice": @{ @"type": @"auto", @"disable_parallel_tool_use": value } } mutableCopy];
		reply = [self post:choice path:@"/v1/messages" status:&status];
		XCTAssertEqual(status, expected, @"disable_parallel_tool_use = %@: %@", value, reply);
		XCTAssertTrue(absent || [reply[@"error"][@"message"] containsString:@"tool_choice.disable_parallel_tool_use"], @"%@", reply);

		NSMutableDictionary *usage = [[self chatBody] mutableCopy];
		usage[@"stream_options"] = @{ @"include_usage": value };
		reply = [self post:usage path:@"/v1/chat/completions" status:&status];
		XCTAssertEqual(status, expected, @"include_usage = %@: %@", value, reply);
		XCTAssertTrue(absent || [reply[@"error"][@"message"] containsString:@"stream_options.include_usage"], @"%@", reply);
	}
	XCTAssertEqual(self.backend.runs, 3u, @"only the null flags ran");
	[self assertStillServes:@"the refused nested flags"];
}

#pragma mark Documented fields

- (void)testChatAndOpenAIRoutesRefuseEveryDocumentedFieldOfAnotherType
{
	[self fuzzPath:@"/v1/chat/completions" base:[self chatBody] fields:@{
		@"model": @"string", @"messages": @"array", @"stream": @"boolean", @"stream_options": @"object", @"n": @"integer",
		@"modalities": @"array", @"audio": @"object", @"tools": @"array", @"tool_choice": @"string|object",
		@"parallel_tool_calls": @"boolean", @"response_format": @"object", @"stop": @"string|array",
		@"reasoning_effort": @"string", @"max_tokens": @"integer", @"max_completion_tokens": @"integer",
		@"temperature": @"number", @"top_p": @"number", @"top_k": @"integer", @"seed": @"integer",
		@"repetition_penalty": @"number", @"repeat_penalty": @"number", @"presence_penalty": @"number",
		@"frequency_penalty": @"number", @"prompt_cache_key": @"string", @"user": @"string", @"logprobs": @"boolean",
		@"top_logprobs": @"integer", @"metadata": @"object", @"store": @"boolean" } anthropic:NO];
	[self fuzzPath:@"/v1/embeddings" base:@{ @"model": @"served", @"input": @"hi" } fields:@{
		@"model": @"string", @"input": @"string|array", @"encoding_format": @"string", @"dimensions": @"integer",
		@"user": @"string" } anthropic:NO];
	[self fuzzPath:@"/v1/audio/speech" base:@{ @"model": @"served", @"input": @"hi" } fields:@{
		@"model": @"string", @"input": @"string", @"voice": @"string|object", @"response_format": @"string",
		@"stream_format": @"string", @"stream": @"boolean", @"speed": @"number", @"language": @"string",
		@"sample_rate": @"integer", @"instructions": @"string" } anthropic:NO];
	[self fuzzPath:@"/v1/images/generations" base:@{ @"model": @"served", @"prompt": @"a cat" } fields:@{
		@"model": @"string", @"prompt": @"string", @"negative_prompt": @"string", @"n": @"integer", @"size": @"string",
		@"response_format": @"string", @"output_format": @"string", @"stream": @"boolean", @"seed": @"integer",
		@"steps": @"integer", @"guidance_scale": @"number", @"aspect_ratio": @"string", @"resolution": @"string",
		@"quality": @"string", @"style": @"string", @"background": @"string", @"output_compression": @"integer",
		@"partial_images": @"integer", @"user": @"string" } anthropic:NO];
}

- (void)testMessagesAndResponsesRoutesRefuseEveryDocumentedFieldOfAnotherType
{
	NSDictionary *messages = @{ @"model": @"served", @"max_tokens": @16, @"messages": @[ @{ @"role": @"user", @"content": @"hi" } ] };
	NSDictionary *anthropic = @{ @"model": @"string", @"messages": @"array", @"system": @"string|array", @"max_tokens": @"integer",
								 @"temperature": @"number", @"top_p": @"number", @"top_k": @"integer", @"stop_sequences": @"array",
								 @"stream": @"boolean", @"tools": @"array", @"tool_choice": @"object", @"thinking": @"object",
								 @"output_config": @"object", @"metadata": @"object", @"service_tier": @"string", @"mcp_servers": @"array" };
	[self fuzzPath:@"/v1/messages" base:messages fields:anthropic anthropic:YES];
	[self fuzzPath:@"/v1/messages/count_tokens" base:messages fields:anthropic anthropic:YES];
	NSDictionary *responses = @{ @"model": @"string", @"input": @"string|array", @"instructions": @"string", @"stream": @"boolean",
								 @"stream_options": @"object", @"tools": @"array", @"tool_choice": @"string|object", @"temperature": @"number",
								 @"top_p": @"number", @"max_output_tokens": @"integer", @"reasoning": @"object", @"text": @"object",
								 @"prompt_cache_key": @"string", @"previous_response_id": @"string", @"conversation": @"string|object",
								 @"store": @"boolean", @"include": @"array", @"background": @"boolean", @"metadata": @"object",
								 @"user": @"string", @"safety_identifier": @"string", @"service_tier": @"string", @"truncation": @"string",
								 @"parallel_tool_calls": @"boolean", @"max_tool_calls": @"integer", @"top_logprobs": @"integer" };
	[self fuzzPath:@"/v1/responses" base:@{ @"model": @"served", @"input": @"hi" } fields:responses anthropic:NO];
	[self fuzzPath:@"/v1/responses/input_tokens" base:@{ @"model": @"served", @"input": @"hi" } fields:responses anthropic:NO];
}

- (void)testNativeAndEstimateRoutesRefuseEveryDocumentedFieldOfAnotherType
{
	NSDictionary *run = @{ @"model": @"served", @"request": @{ @"inputs": @{ @"prompt": @"hi" } } };
	[self fuzzPath:@"/v1/inferkit/run" base:run fields:@{ @"model": @"string", @"request": @"object", @"stream": @"boolean" } anthropic:NO];
	[self fuzzPath:@"/v1/inferkit/estimate" base:@{ @"model": @"served" } fields:@{ @"model": @"string", @"options": @"object" } anthropic:NO];
	for (NSString *field in @[ @"inputs", @"parameters", @"outputModality" ]) {
		NSMutableDictionary *request = [@{ @"inputs": @{ @"prompt": @"hi" } } mutableCopy];
		request[field] = [field isEqualToString:@"outputModality"] ? @"text" : (id)@[ @1 ];
		NSInteger status = 0;
		NSDictionary *reply = [self post:@{ @"model": @"served", @"request": request } path:@"/v1/inferkit/run" status:&status];
		XCTAssertEqual(status, 400, @"request.%@", field);
		XCTAssertTrue([reply[@"error"][@"message"] containsString:[@"request." stringByAppendingString:field]], @"%@", reply);
	}
}

- (void)testAModelThatIsNotAStringIsRefusedAndANullOneIsAbsent
{
	for (id model in @[ @5, @[ @"served" ], @{ @"id": @"served" }, @YES ]) {
		NSMutableDictionary *body = [[self chatBody] mutableCopy];
		body[@"model"] = model;
		NSInteger status = 0;
		NSDictionary *reply = [self post:body path:@"/v1/chat/completions" status:&status];
		XCTAssertEqual(status, 400, @"model = %@", model);
		XCTAssertTrue([reply[@"error"][@"message"] containsString:@"model"], @"%@", reply);
	}
	XCTAssertEqual(self.backend.runs, 0u);
	NSMutableDictionary *unnamed = [[self chatBody] mutableCopy];
	[unnamed removeObjectForKey:@"model"];
	NSInteger status = 0;
	[self post:unnamed path:@"/v1/chat/completions" status:&status];
	XCTAssertEqual(status, 200, @"a request that names no model still runs on the only one");
	unnamed[@"model"] = NSNull.null;
	[self post:unnamed path:@"/v1/chat/completions" status:&status];
	XCTAssertEqual(status, 200, @"a null model names none");
}

#pragma mark Nested values

- (void)testMessagesAndItemsOfAnotherTypeAreRefused
{
	NSDictionary<NSString *, NSArray<NSDictionary *> *> *cases = @{
		@"/v1/chat/completions": @[
			@{ @"model": @"served", @"messages": @[ @{ @"role": @"user", @"content": @5 } ] },
			@{ @"model": @"served", @"messages": @[ @{ @"role": @[ @"user" ], @"content": @"hi" } ] },
			@{ @"model": @"served", @"messages": @[ @{ @"role": @"user", @"content": @[ @{ @"type": @"text", @"text": @5 } ] } ] },
			@{ @"model": @"served", @"messages": @[ @{ @"role": @"assistant", @"content": NSNull.null,
													   @"tool_calls": @[ @{ @"id": @"c", @"type": @"function",
																			@"function": @{ @"name": @"f", @"arguments": @{} } } ] } ] } ],
		@"/v1/messages": @[
			@{ @"model": @"served", @"max_tokens": @16, @"system": @[ @{ @"type": @"text", @"text": @5 } ],
			   @"messages": @[ @{ @"role": @"user", @"content": @"hi" } ] },
			@{ @"model": @"served", @"max_tokens": @16, @"messages": @[ @{ @"role": @"user", @"content": @[ @{ @"type": @"text", @"text": @[] } ] } ] },
			@{ @"model": @"served", @"max_tokens": @16, @"stop_sequences": @[ @5 ], @"messages": @[ @{ @"role": @"user", @"content": @"hi" } ] },
			@{ @"model": @"served", @"max_tokens": @16, @"thinking": @{ @"type": @"enabled", @"budget_tokens": @"lots" },
			   @"messages": @[ @{ @"role": @"user", @"content": @"hi" } ] } ],
		@"/v1/responses": @[
			@{ @"model": @"served", @"input": @[ @{ @"type": @"message", @"role": @"user", @"content": @5 } ] },
			@{ @"model": @"served", @"input": @[ @{ @"type": @"function_call", @"call_id": @"c", @"name": @"f", @"arguments": @{} } ] },
			@{ @"model": @"served", @"input": @"hi", @"reasoning": @{ @"effort": @3 } },
			@{ @"model": @"served", @"input": @"hi", @"text": @{ @"format": @{ @"type": @"json_schema", @"schema": @"{}" } } } ] };
	for (NSString *path in cases) {
		for (NSDictionary *body in cases[path]) {
			NSInteger status = 0;
			NSDictionary *reply = [self post:body path:path status:&status];
			XCTAssertEqual(status, 400, @"%@ %@: %@", path, body, reply);
		}
	}
	XCTAssertEqual(self.backend.runs, 0u);
	NSInteger status = 0;
	[self post:@{ @"model": @"served", @"input": @[ @{ @"type": @"reasoning", @"id": NSNull.null, @"summary": @[], @"content": NSNull.null },
													 @{ @"type": @"message", @"role": @"user", @"content": @"hi" } ] }
		  path:@"/v1/responses" status:&status];
	XCTAssertEqual(status, 200, @"a reasoning item, which the server skips, is not read");
	[self assertStillServes:@"the refused messages"];
}

- (void)testAContractParameterOfAnotherKindIsRefusedBeforeItReachesTheBackend
{
	NSMutableDictionary *chat = [[self chatBody] mutableCopy];
	chat[NFKParameterTopP] = @"high";
	NSInteger status = 0;
	NSDictionary *reply = [self post:chat path:@"/v1/chat/completions" status:&status];
	XCTAssertEqual(status, 400);
	XCTAssertTrue([reply[@"error"][@"message"] containsString:NFKParameterTopP], @"%@", reply);

	NSDictionary *cases = @{ @"inputs": @{ NFKInputPrompt: @5 },
							 @"parameters": @{ NFKParameterTemperature: @"hot" },
							 @"stop": @{ NFKParameterStopSequences: @[ @"END", @5 ] } };
	for (NSString *name in cases) {
		NSDictionary *request = [name isEqualToString:@"inputs"]
			? @{ @"inputs": cases[name] } : @{ @"inputs": @{ NFKInputPrompt: @"hi" }, @"parameters": cases[name] };
		for (NSNumber *stream in @[ @NO, @YES ]) {
			reply = [self post:@{ @"model": @"served", @"request": request, @"stream": stream } path:@"/v1/inferkit/run" status:&status];
			XCTAssertEqual(status, 400, @"%@, stream %@: %@", name, stream, reply);
		}
	}
	XCTAssertEqual(self.backend.runs, 0u);
	NSDictionary *nulled = @{ @"inputs": @{ NFKInputPrompt: @"hi" }, @"parameters": @{ NFKParameterTemperature: NSNull.null, @"own": NSNull.null } };
	reply = [self post:@{ @"model": @"served", @"request": nulled } path:@"/v1/inferkit/run" status:&status];
	XCTAssertEqual(status, 200, @"a null contract value is absent: %@", reply);
	XCTAssertNil([self.backend.lastRequest parameterForKey:NFKParameterTemperature], @"the backend meets no NSNull under a contract key");
	XCTAssertEqualObjects([self.backend.lastRequest parameterForKey:@"own"], NSNull.null, @"a key outside the contract keeps the null the wire carries");
	[self assertStillServes:@"the refused contract values"];
}

#pragma mark Form fields

/*! A multipart/form-data body of fields, with a file part when file is set, and its Content-Type. */
- (NSData *)formWithFields:(NSDictionary<NSString *, NSString *> *)fields file:(nullable NSData *)file
				  fileName:(NSString *)fileName contentType:(NSString * _Nonnull * _Nonnull)outContentType
{
	NSString *boundary = NSUUID.UUID.UUIDString;
	NSMutableData *body = [NSMutableData data];
	[fields enumerateKeysAndObjectsUsingBlock:^(NSString *name, NSString *value, BOOL *stop) {
		[body appendData:[[NSString stringWithFormat:@"--%@\r\nContent-Disposition: form-data; name=\"%@\"\r\n\r\n%@\r\n", boundary, name, value]
						  dataUsingEncoding:NSUTF8StringEncoding]];
	}];
	if (file != nil) {
		[body appendData:[[NSString stringWithFormat:@"--%@\r\nContent-Disposition: form-data; name=\"%@\"; filename=\"%@\"\r\n"
						   "Content-Type: application/octet-stream\r\n\r\n", boundary, fileName, fileName]
						  dataUsingEncoding:NSUTF8StringEncoding]];
		[body appendData:file];
		[body appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
	}
	[body appendData:[[NSString stringWithFormat:@"--%@--\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
	*outContentType = [@"multipart/form-data; boundary=" stringByAppendingString:boundary];
	return body;
}

- (void)testFormFieldsThatDoNotReadAsTheirKindAreRefused
{
	NSString *contentType = nil;
	NSData *transcription = [self formWithFields:@{ @"model": @"served", @"temperature": @"warm" }
											file:[@"RIFF....WAVE" dataUsingEncoding:NSUTF8StringEncoding] fileName:@"file" contentType:&contentType];
	NSInteger status = 0;
	NSDictionary *reply = [self send:@"POST" path:@"/v1/audio/transcriptions" data:transcription headers:@{ @"Content-Type": contentType } status:&status];
	XCTAssertEqual(status, 400, @"%@", reply);
	XCTAssertTrue([reply[@"error"][@"message"] containsString:@"temperature"], @"%@", reply);

	NSData *edit = [self formWithFields:@{ @"model": @"served", @"prompt": @"a hat", @"n": @"many" } file:nil fileName:@"image" contentType:&contentType];
	reply = [self send:@"POST" path:@"/v1/images/edits" data:edit headers:@{ @"Content-Type": contentType } status:&status];
	XCTAssertEqual(status, 400, @"%@", reply);
	XCTAssertEqual(self.backend.runs, 0u);
	[self assertStillServes:@"the refused form fields"];
}

#pragma mark Content-Type and Origin

/*! The status of a POST written to the listener by hand, so its Host and its other headers are the
	test's own; a URL session replaces Host and adds a Content-Type to a body without one. 0 when no
	reply arrives within the socket's 10 s deadline. */
- (NSInteger)statusOfHandWrittenPost:(NSString *)path host:(NSString *)host
							 headers:(NSDictionary<NSString *, NSString *> *)headers body:(NSData *)body
{
	int descriptor = socket(AF_INET, SOCK_STREAM, 0);
	XCTAssertGreaterThanOrEqual(descriptor, 0);
	struct timeval timeout = { 10, 0 };
	setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
	setsockopt(descriptor, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
	struct sockaddr_in address = { 0 };
	address.sin_len = sizeof(address);
	address.sin_family = AF_INET;
	address.sin_port = htons(self.server.listeningPort);
	address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	if (connect(descriptor, (struct sockaddr *)&address, sizeof(address)) != 0) {
		close(descriptor);
		return 0;
	}
	NSMutableString *head = [NSMutableString stringWithFormat:@"POST %@ HTTP/1.1\r\nHost: %@\r\nConnection: close\r\n", path, host];
	[headers enumerateKeysAndObjectsUsingBlock:^(NSString *field, NSString *value, BOOL *stop) {
		[head appendFormat:@"%@: %@\r\n", field, value];
	}];
	[head appendFormat:@"Content-Length: %lu\r\n\r\n", (unsigned long)body.length];
	NSMutableData *message = [[head dataUsingEncoding:NSUTF8StringEncoding] mutableCopy];
	[message appendData:body];
	write(descriptor, message.bytes, message.length);
	char reply[64] = { 0 };
	ssize_t received = recv(descriptor, reply, sizeof(reply) - 1, 0);
	close(descriptor);
	NSString *line = received > 0 ? [[NSString alloc] initWithBytes:reply length:(NSUInteger)received encoding:NSASCIIStringEncoding] : nil;
	NSArray<NSString *> *words = [line componentsSeparatedByString:@" "];
	return words.count > 1 ? words[1].integerValue : 0;
}

- (NSInteger)statusOfPostWithoutContentType:(NSString *)path body:(NSData *)body
{
	NSString *host = [NSString stringWithFormat:@"localhost:%u", (unsigned)self.server.listeningPort];
	return [self statusOfHandWrittenPost:path host:host headers:@{} body:body];
}

- (void)testAPostWithoutAContentTypeIsRead
{
	NSData *chat = [NSJSONSerialization dataWithJSONObject:[self chatBody] options:0 error:NULL];
	XCTAssertEqual([self statusOfPostWithoutContentType:@"/v1/chat/completions" body:chat], 200,
				   @"a client that omits the type is served; a page's post carries an Origin");
	NSData *ping = [NSJSONSerialization dataWithJSONObject:@{ @"jsonrpc": @"2.0", @"id": @1, @"method": @"ping" } options:0 error:NULL];
	XCTAssertEqual([self statusOfPostWithoutContentType:@"/mcp" body:ping], 200);
	XCTAssertEqual(self.backend.runs, 1u);
}

- (void)testAJSONBodyIsReadWhateverItsContentType
{
	NSDictionary *messages = @{ @"model": @"served", @"max_tokens": @16, @"messages": @[ @{ @"role": @"user", @"content": @"hi" } ] };
	NSDictionary<NSString *, NSDictionary *> *bodies = @{ @"/v1/chat/completions": [self chatBody], @"/v1/messages": messages,
														  @"/v1/responses": @{ @"model": @"served", @"input": @"hi" },
														  @"/v1/inferkit/run": @{ @"model": @"served", @"request": @{ @"inputs": @{ NFKInputPrompt: @"hi" } } },
														  @"/mcp": @{ @"jsonrpc": @"2.0", @"id": @1, @"method": @"ping" } };
	NSArray<NSString *> *types = @[ @"text/plain;charset=UTF-8", @"application/x-www-form-urlencoded", @"multipart/form-data; boundary=x",
									@"application/json; charset=utf-8", @"application/vnd.inferkit+json" ];
	for (NSString *type in types) {
		for (NSString *path in bodies) {
			NSInteger status = 0;
			NSData *data = [NSJSONSerialization dataWithJSONObject:bodies[path] options:0 error:NULL];
			id reply = [self send:@"POST" path:path data:data headers:@{ @"Content-Type": type } status:&status];
			XCTAssertEqual(status, 200, @"%@ under %@, as curl -d sends it: %@", path, type, reply);
		}
	}
	XCTAssertEqual(self.backend.runs, 4u * types.count);
	NSData *notJSON = [@"model=served&prompt=hi" dataUsingEncoding:NSUTF8StringEncoding];
	for (NSString *path in bodies) {
		NSInteger status = 0;
		[self send:@"POST" path:path data:notJSON headers:@{ @"Content-Type": @"application/x-www-form-urlencoded" } status:&status];
		XCTAssertEqual(status, 400, @"%@: a body that is not JSON is the route's usual refusal", path);
	}
}

- (void)testABrowserPageOnAnotherSiteIsRefusedOnEveryRoute
{
	self.server.apiKey = @"secret";
	NSData *data = [NSJSONSerialization dataWithJSONObject:[self chatBody] options:0 error:NULL];
	NSDictionary *json = @{ @"Content-Type": @"application/json" };
	for (NSString *origin in @[ @"https://attacker.example", @"null", @"http://127.0.0.1.attacker.example" ]) {
		NSMutableDictionary *headers = [json mutableCopy];
		headers[@"Origin"] = origin;
		NSInteger status = 0;
		[self send:@"POST" path:@"/v1/chat/completions" data:data headers:headers status:&status];
		XCTAssertEqual(status, 403, @"%@", origin);
		headers[@"Authorization"] = @"Bearer secret";
		[self send:@"POST" path:@"/v1/chat/completions" data:data headers:headers status:&status];
		XCTAssertEqual(status, 403, @"a key does not admit a page from %@", origin);
		[self send:@"GET" path:@"/v1/models" data:nil headers:headers status:&status];
		XCTAssertEqual(status, 403, @"%@ reads no model list", origin);
		[self send:@"GET" path:@"/v1/inferkit/status" data:nil headers:headers status:&status];
		XCTAssertEqual(status, 403, @"%@ reads no status", origin);
	}
	NSInteger status = 0;
	NSDictionary *reply = [self send:@"POST" path:@"/v1/messages" data:[NSJSONSerialization dataWithJSONObject:@{ @"model": @"served", @"max_tokens": @16,
																												   @"messages": @[ @{ @"role": @"user", @"content": @"hi" } ] }
																										 options:0 error:NULL]
							 headers:@{ @"Content-Type": @"application/json", @"Origin": @"https://attacker.example" } status:&status];
	XCTAssertEqual(status, 403);
	XCTAssertEqualObjects(reply[@"error"][@"type"], @"permission_error", @"the Messages route refuses in Anthropic's shape");
	XCTAssertEqual(self.backend.runs, 0u);

	NSMutableDictionary *local = [json mutableCopy];
	local[@"Origin"] = @"http://localhost:3000";
	[self send:@"POST" path:@"/v1/chat/completions" data:data headers:local status:&status];
	XCTAssertEqual(status, 200, @"a page served from this machine passes");
	[self send:@"POST" path:@"/v1/chat/completions" data:data headers:json status:&status];
	XCTAssertEqual(status, 200, @"a local agent sends no Origin and needs no key on loopback");
}

- (void)testAllowedOriginsAdmitAPageBehindAProxyThatRewritesHost
{
	self.server.apiKey = @"secret";
	NSData *chat = [NSJSONSerialization dataWithJSONObject:[self chatBody] options:0 error:NULL];
	NSDictionary *key = @{ @"Authorization": @"Bearer secret" };
	NSMutableDictionary *sameHost = [key mutableCopy];
	sameHost[@"Origin"] = @"https://models.example.com";
	XCTAssertEqual([self statusOfHandWrittenPost:@"/v1/chat/completions" host:@"models.example.com" headers:sameHost body:chat], 200,
				   @"without allowedOrigins a page on the host the request names passes");
	NSString *upstream = [NSString stringWithFormat:@"127.0.0.1:%u", (unsigned)self.server.listeningPort];
	XCTAssertEqual([self statusOfHandWrittenPost:@"/v1/chat/completions" host:upstream headers:sameHost body:chat], 403,
				   @"a proxy that rewrites Host to its upstream makes the same page look foreign");

	self.server.allowedOrigins = [NSSet setWithObject:@"https://models.example.com"];
	XCTAssertEqual([self statusOfHandWrittenPost:@"/v1/chat/completions" host:upstream headers:sameHost body:chat], 200,
				   @"the listed origin passes behind the proxy");
	for (NSString *origin in @[ @"HTTPS://Models.Example.com/", @"https://models.example.com" ]) {
		NSInteger status = 0;
		[self send:@"POST" path:@"/v1/chat/completions" data:chat headers:@{ @"Content-Type": @"application/json", @"Origin": origin } status:&status];
		XCTAssertEqual(status, 200, @"%@ compares without case or a trailing slash", origin);
	}
	NSMutableDictionary *unlisted = [key mutableCopy];
	unlisted[@"Origin"] = @"https://models.example.com:8443";
	XCTAssertEqual([self statusOfHandWrittenPost:@"/v1/chat/completions" host:@"models.example.com:8443" headers:unlisted body:chat], 403,
				   @"with allowedOrigins set, a same-host origin it does not list is refused");
	NSInteger status = 0;
	[self send:@"POST" path:@"/v1/chat/completions" data:chat headers:@{ @"Content-Type": @"application/json", @"Origin": @"http://localhost:5173" } status:&status];
	XCTAssertEqual(status, 200, @"a page on this machine passes either way");

	NSDictionary *ping = @{ @"jsonrpc": @"2.0", @"id": @1, @"method": @"ping" };
	NSData *pingData = [NSJSONSerialization dataWithJSONObject:ping options:0 error:NULL];
	[self send:@"POST" path:@"/mcp" data:pingData headers:@{ @"Origin": @"https://models.example.com" } status:&status];
	XCTAssertEqual(status, 200, @"/mcp applies the same set");
	NSDictionary *refusal = [self send:@"POST" path:@"/mcp" data:pingData headers:@{ @"Origin": @"https://attacker.example" } status:&status];
	XCTAssertEqual(status, 403);
	XCTAssertEqualObjects(refusal[@"error"][@"code"], @(-32600), @"the MCP transport's JSON-RPC shape");
	XCTAssertEqualObjects(refusal[@"error"][@"data"][@"inferkit_domain"], NFKInferenceServerErrorDomain);
	XCTAssertEqualObjects(refusal[@"error"][@"data"][@"inferkit_code"], @(NFKInferenceServerErrorOriginRefused));
}

- (void)testAnIntegerPastTheLargestLongIsRefused
{
	NSArray *refused = @[ @(9223372036854775808.0), @(9223372036854775808ULL), @(1.8e19) ];
	for (NSNumber *seed in refused) {
		NSMutableDictionary *body = [[self chatBody] mutableCopy];
		body[@"seed"] = seed;
		NSInteger status = 0;
		NSDictionary *reply = [self post:body path:@"/v1/chat/completions" status:&status];
		XCTAssertEqual(status, 400, @"seed %@ overflows a long long: %@", seed, reply);
	}
	XCTAssertEqual(self.backend.runs, 0u);
	for (NSNumber *seed in @[ @(LLONG_MIN), @(LLONG_MAX), @(9223372036854774784.0) ]) {
		NSMutableDictionary *body = [[self chatBody] mutableCopy];
		body[@"seed"] = seed;
		NSInteger status = 0;
		[self post:body path:@"/v1/chat/completions" status:&status];
		XCTAssertEqual(status, 200, @"seed %@ fits", seed);
	}
}

#pragma mark Exceptions

- (void)testAnExceptionAnswers500AndTheServerKeepsServing
{
	self.backend.raises = YES;
	NSInteger status = 0;
	NSDictionary *reply = [self post:[self chatBody] path:@"/v1/chat/completions" status:&status];
	XCTAssertEqual(status, 500, @"%@", reply);
	XCTAssertTrue([reply[@"error"][@"message"] containsString:NSInvalidArgumentException], @"%@", reply);
	self.backend.raises = NO;
	[self assertStillServes:@"a backend that raised"];

	self.backend.result = [NFKInferenceResult resultWithOutputs:@{ NFKOutputEmbedding: @[ @(NAN) ] }];
	[self post:@{ @"model": @"served", @"input": @"hi" } path:@"/v1/embeddings" status:&status];
	XCTAssertEqual(status, 500, @"a reply JSON cannot carry is a server error");
	self.backend.result = nil;
	[self assertStillServes:@"a reply JSON cannot carry"];

	NSError *error = nil;
	XCTAssertTrue([self.server addRouteForPathPrefix:@"/app" handler:^(NFKServerRouteRequest *request, NFKServerRouteResponse *response) {
		[NSException raise:NSRangeException format:@"the app route raises"];
	} error:&error], @"%@", error);
	[self send:@"GET" path:@"/app/x" data:nil headers:nil status:&status];
	XCTAssertEqual(status, 500);
	[self assertStillServes:@"an app route that raised"];
}

#pragma mark MCP

- (nullable id)postMCP:(id)body status:(NSInteger *)outStatus
{
	return [self post:body path:@"/mcp" status:outStatus];
}

- (void)testMCPMessagesOfAnotherTypeAreRefused
{
	NSArray<NSDictionary *> *malformed = @[ @{ @"jsonrpc": @"2.0", @"id": @{ @"n": @1 }, @"method": @"ping" },
											@{ @"jsonrpc": @"2.0", @"id": @1, @"method": @5 },
											@{ @"jsonrpc": @"2.0", @"id": @1, @"method": @"tools/list", @"params": @[ @1 ] },
											@{ @"jsonrpc": @"2.0", @"id": @1, @"method": @"tools/call", @"params": @{ @"name": @5 } },
											@{ @"jsonrpc": @"2.0", @"id": @1, @"method": @"tools/call", @"params": @{ @"name": @"x", @"arguments": @"{}" } },
											@{ @"jsonrpc": @"2.0", @"id": @1, @"method": @"initialize", @"params": @{ @"clientInfo": @"me" } } ];
	for (NSDictionary *message in malformed) {
		NSInteger status = 0;
		NSDictionary *reply = [self postMCP:message status:&status];
		XCTAssertEqual(status, 400, @"%@", message);
		XCTAssertNotNil(reply[@"error"][@"code"], @"%@", reply);
	}
	NSInteger status = 0;
	NSDictionary *ping = [self postMCP:@{ @"jsonrpc": @"2.0", @"id": @1, @"method": @"ping", @"params": NSNull.null } status:&status];
	XCTAssertEqual(status, 200, @"a null params is absent: %@", ping);
	NSDictionary *listed = [self postMCP:@{ @"jsonrpc": @"2.0", @"id": @1, @"method": @"initialize",
											@"params": @{ @"protocolVersion": NSNull.null, @"clientInfo": NSNull.null } } status:&status];
	XCTAssertEqual(status, 200, @"a null field in params is absent: %@", listed);
	NSArray *replies = [self postMCP:@[ malformed[4], @{ @"jsonrpc": @"2.0", @"id": @2, @"method": @"ping" } ] status:&status];
	XCTAssertEqual(status, 200, @"a batch answers each message");
	XCTAssertEqualObjects(replies[0][@"error"][@"code"], @(-32602));
	XCTAssertEqualObjects(replies[1][@"result"], @{});
}

/*! Sends count concurrent tools/call requests for tool to server's /mcp, and asserts that each answers
	with a result before a 90 s deadline. */
- (void)callTool:(NSString *)tool concurrently:(NSUInteger)count onServer:(NFKInferenceServer *)server
{
	NSURLSessionConfiguration *configuration = NSURLSessionConfiguration.ephemeralSessionConfiguration;
	configuration.HTTPMaximumConnectionsPerHost = 128;
	configuration.timeoutIntervalForRequest = 120;
	NSURLSession *session = [NSURLSession sessionWithConfiguration:configuration];
	[self addTeardownBlock:^{
		[session invalidateAndCancel];
	}];
	dispatch_group_t group = dispatch_group_create();
	NSMutableArray *replies = [NSMutableArray array];
	for (NSUInteger index = 0; index < count; index++) {
		NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:server.localMCPURL];
		request.HTTPMethod = @"POST";
		[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
		request.HTTPBody = [NSJSONSerialization dataWithJSONObject:@{ @"jsonrpc": @"2.0", @"id": @(index), @"method": @"tools/call",
																	  @"params": @{ @"name": tool, @"arguments": @{ NFKInputPrompt: @"hi" } } }
														   options:0 error:NULL];
		dispatch_group_enter(group);
		[[session dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *taskError) {
			id reply = data.length > 0 ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
			@synchronized (replies) {
				[replies addObject:reply ?: (id)(taskError.localizedDescription ?: @"no reply")];
			}
			dispatch_group_leave(group);
		}] resume];
	}
	long waited = dispatch_group_wait(group, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(90 * NSEC_PER_SEC)));
	XCTAssertEqual(waited, 0, @"every call answers before the deadline; a blocked worker per call starves the runs");
	@synchronized (replies) {
		XCTAssertEqual(replies.count, count);
		for (id reply in replies) {
			XCTAssertTrue([reply isKindOfClass:NSDictionary.class], @"%@", reply);
			XCTAssertEqualObjects(reply[@"result"][@"isError"], @NO, @"%@", reply);
		}
	}
}

- (void)testMoreConcurrentModelToolCallsThanTheThreadPoolAllAnswer
{
	self.backend.pause = 0.01;
	NSError *error = nil;
	NFKBackendTool *tool = [self.server addToolForModelName:@"served" outputModality:NFKModalityText purpose:nil error:&error];
	XCTAssertNotNil(tool, @"%@", error);
	[self callTool:tool.name concurrently:100 onServer:self.server];
	XCTAssertEqual(self.backend.runs, 100u);
}

- (void)testAModelToolOfAnotherServerRunsInThatServerWithoutHoldingAThread
{
	NFKRequestChecksBackend *other = [[NFKRequestChecksBackend alloc] init];
	other.pause = 0.01;
	NFKInferenceServer *owner = [[NFKInferenceServer alloc] init];
	owner.port = 0;
	owner.loopbackOnly = YES;
	owner.advertisesService = NO;
	[owner addBackend:other forModelName:@"elsewhere"];
	NSError *error = nil;
	XCTAssertTrue([owner startWithError:&error], @"%@", error);
	[self addTeardownBlock:^{
		[owner stop];
	}];
	NFKBackendTool *tool = [owner addToolForModelName:@"elsewhere" outputModality:NFKModalityText purpose:nil error:&error];
	XCTAssertNotNil(tool, @"%@", error);
	[self.server addTool:tool];
	[self callTool:tool.name concurrently:100 onServer:self.server];
	XCTAssertEqual(other.runs, 100u, @"the runs happen in the owning server's queue");
	XCTAssertEqual(self.backend.runs, 0u);
}

- (void)testABatchOfModelToolCallsAnswersInOrder
{
	NSError *error = nil;
	NFKBackendTool *tool = [self.server addToolForModelName:@"served" outputModality:NFKModalityText purpose:nil error:&error];
	XCTAssertNotNil(tool, @"%@", error);
	NSMutableArray *batch = [NSMutableArray array];
	for (NSUInteger index = 0; index < 3; index++) {
		[batch addObject:@{ @"jsonrpc": @"2.0", @"id": @(index), @"method": @"tools/call",
							@"params": @{ @"name": tool.name, @"arguments": @{ NFKInputPrompt: @"hi" } } }];
	}
	[batch addObject:@{ @"jsonrpc": @"2.0", @"id": @3, @"method": @"tools/call", @"params": @{ @"name": tool.name, @"arguments": @{} } }];
	NSInteger status = 0;
	NSArray *replies = [self postMCP:batch status:&status];
	XCTAssertEqual(status, 200);
	XCTAssertEqual(replies.count, 4u);
	for (NSUInteger index = 0; index < replies.count; index++) {
		XCTAssertEqualObjects(replies[index][@"id"], @(index));
	}
	XCTAssertEqualObjects(replies[0][@"result"][@"isError"], @NO);
	XCTAssertEqualObjects(replies[3][@"result"][@"isError"], @YES, @"a call missing its required argument is not run");
	XCTAssertEqual(self.backend.runs, 3u);
}

@end
