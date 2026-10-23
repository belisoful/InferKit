//
//  NFKServedGeminiTests.m
//  InferKitTests
//
//  Google's generateContent API served by NFKInferenceServer, over real HTTP on a loopback port: the
//  request in the contract's shape, the GenerateContentResponse and its event stream, function calls
//  both ways, token counting, the model list, keys in Google's forms, errors in Google's shape, and
//  NFKGeminiGenerateContentBackend pointed at the server.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

/*! A synchronous backend whose reply a test writes, recording what reached it, and counting a request's
	tokens as the characters in its messages' text. */
@interface NFKGeminiRouteBackend : NSObject <NFKInferenceBackend>
@property (atomic, strong, nullable) NFKInferenceRequest *lastRequest;
@property (nonatomic, copy, nullable) NFKInferenceResult * _Nullable (^reply)(NFKInferenceRequest *request, NSError **error);
@end

@implementation NFKGeminiRouteBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"gemini-route-test";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	self.lastRequest = request;
	return self.reply != nil ? self.reply(request, error) : [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"ok" }];
}

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

/*! A backend whose job streams cumulative partials, then finishes with the final result a test writes. */
@interface NFKGeminiRouteStreamingBackend : NFKGeminiRouteBackend
@property (nonatomic, copy) NSArray<NFKInferenceResult *> *partials;
@property (nonatomic, strong, nullable) NFKInferenceResult *finalResult;
@end

@implementation NFKGeminiRouteStreamingBackend

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	self.lastRequest = request;
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	NSArray<NFKInferenceResult *> *partials = self.partials;
	NFKInferenceResult *finalResult = self.finalResult;
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		for (NFKInferenceResult *partial in partials) {
			[job reportProgress:-1.0 partialResult:partial];
			[NSThread sleepForTimeInterval:0.02];
		}
		[job finishWithResult:finalResult];
	});
	return job;
}

@end

@interface NFKServedGeminiTests : XCTestCase
@end

@implementation NFKServedGeminiTests

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

- (nullable id)JSONOf:(nullable NSString *)text
{
	return text != nil ? [NSJSONSerialization JSONObjectWithData:[text dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL] : nil;
}

- (nullable NSDictionary *)post:(NSString *)path body:(NSDictionary *)body server:(NFKInferenceServer *)server status:(NSInteger *)outStatus
{
	return [self JSONOf:[self send:@"POST" path:path body:body headers:nil server:server status:outStatus]];
}

/*! The data of each event in a server-sent event stream. */
- (NSArray<NSDictionary *> *)eventsIn:(NSString *)stream
{
	NSMutableArray<NSDictionary *> *events = [NSMutableArray array];
	for (NSString *block in [stream componentsSeparatedByString:@"\n\n"]) {
		if ([block hasPrefix:@"data: "]) {
			[events addObject:[self JSONOf:[block substringFromIndex:6]]];
		}
	}
	return events;
}

- (NFKGeminiGenerateContentBackend *)clientFor:(NFKInferenceServer *)server model:(NSString *)model
{
	NSURL *base = [NSURL URLWithString:[NSString stringWithFormat:@"http://localhost:%u/v1beta/models", (unsigned)server.listeningPort]];
	NFKGeminiGenerateContentBackend *client = [NFKGeminiGenerateContentBackend backendWithEndpointURL:base];
	client.modelName = model;
	return client;
}

#pragma mark Requests

- (void)testAGenerateContentBodyReachesTheBackendInTheContractShape
{
	NFKGeminiRouteBackend *backend = [[NFKGeminiRouteBackend alloc] init];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend } configuration:nil];
	NSDictionary *body = @{
		@"systemInstruction": @{ @"parts": @[ @{ @"text": @"Be brief." } ] },
		@"contents": @[
			@{ @"role": @"user", @"parts": @[ @{ @"text": @"Weather in Paris?" } ] },
			@{ @"role": @"model", @"parts": @[ @{ @"text": @"thinking", @"thought": @YES },
											   @{ @"functionCall": @{ @"name": @"get_weather", @"args": @{ @"city": @"Paris" } } } ] },
			@{ @"role": @"user", @"parts": @[ @{ @"functionResponse": @{ @"name": @"get_weather", @"response": @{ @"output": @"18 C" } } } ] },
		],
		@"tools": @[ @{ @"functionDeclarations": @[ @{ @"name": @"get_weather", @"description": @"The weather now.",
														@"parameters": @{ @"type": @"OBJECT",
																		  @"properties": @{ @"city": @{ @"type": @"STRING" } } } } ] },
					 @{ @"googleSearch": @{} } ],
		@"toolConfig": @{ @"functionCallingConfig": @{ @"mode": @"ANY" } },
		@"generationConfig": @{ @"maxOutputTokens": @64, @"temperature": @0.2, @"topP": @0.9, @"topK": @20, @"stopSequences": @[ @"END" ],
								@"thinkingConfig": @{ @"thinkingBudget": @512, @"includeThoughts": @YES } },
	};
	NSInteger status = 0;
	NSDictionary *reply = [self post:@"/v1beta/models/served:generateContent" body:body server:server status:&status];
	XCTAssertEqual(status, 200, @"%@", reply);
	NFKInferenceRequest *received = backend.lastRequest;
	XCTAssertEqual(received.messages.count, 4u);
	XCTAssertEqualObjects(received.messages[0], (@{ @"role": @"system", @"content": @"Be brief." }));
	NSDictionary *call = [NFKToolMessages toolCallsOfAssistantMessage:received.messages[2]].firstObject;
	XCTAssertEqualObjects(call[@"name"], @"get_weather");
	XCTAssertEqualObjects(call[@"arguments"], @{ @"city": @"Paris" });
	XCTAssertEqualObjects([NFKToolMessages textOfToolMessage:received.messages[3]], @"18 C");
	XCTAssertEqualObjects(received.messages[3][@"tool_call_id"], call[@"id"], @"a response without an id answers its name's call");
	NSDictionary *tool = [[received parameterForKey:NFKParameterTools] firstObject];
	XCTAssertEqual([[received parameterForKey:NFKParameterTools] count], 1u, @"Google's own tools are left out");
	XCTAssertEqualObjects(tool[@"parameters"][@"type"], @"object");
	XCTAssertEqualObjects(tool[@"parameters"][@"properties"][@"city"][@"type"], @"string");
	XCTAssertEqualObjects([received parameterForKey:@"tool_choice"], @"required");
	XCTAssertEqualObjects([received parameterForKey:NFKParameterMaxTokens], @64);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterTopK], @20);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterStopSequences], (@[ @"END" ]));
	XCTAssertEqualObjects([received parameterForKey:NFKParameterReasoningEffort], @"512");
}

- (void)testAJSONResponseMimeTypeAsksForTheSchema
{
	NFKGeminiRouteBackend *backend = [[NFKGeminiRouteBackend alloc] init];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend } configuration:nil];
	NSDictionary *body = @{ @"contents": @[ @{ @"parts": @[ @{ @"text": @"Next speaker?" } ] } ],
							@"generationConfig": @{ @"responseMimeType": @"application/json",
													@"responseJsonSchema": @{ @"type": @"object" } } };
	NSInteger status = 0;
	[self post:@"/v1beta/models/served:generateContent" body:body server:server status:&status];
	XCTAssertEqual(status, 200);
	XCTAssertEqualObjects([backend.lastRequest parameterForKey:NFKParameterJSONSchema], @{ @"type": @"object" });
	XCTAssertEqualObjects(backend.lastRequest.prompt, @"Next speaker?", @"a lone user turn is also the prompt");
}

#pragma mark Replies

- (void)testTheReplyCarriesThoughtsTextCallsTheFinishReasonAndTheUsage
{
	NFKGeminiRouteBackend *backend = [[NFKGeminiRouteBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		return [NFKInferenceResult resultWithOutputs:@{
			NFKOutputText: @"Checking.", NFKOutputReasoning: @"I should look.",
			NFKOutputToolCalls: @[ @{ @"id": @"c1", @"name": @"get_weather", @"arguments": @{ @"city": @"Paris" } } ],
			NFKOutputUsage: @{ NFKUsageInputTokens: @12, NFKUsageOutputTokens: @9, NFKUsageReasoningTokens: @4 } }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend } configuration:nil];
	NSDictionary *body = @{ @"contents": @[ @{ @"role": @"user", @"parts": @[ @{ @"text": @"Weather?" } ] } ],
							@"generationConfig": @{ @"thinkingConfig": @{ @"includeThoughts": @YES } } };
	NSInteger status = 0;
	NSDictionary *reply = [self post:@"/v1beta/models/models/served:generateContent" body:body server:server status:&status];
	XCTAssertEqual(status, 200, @"%@", reply);
	NSDictionary *candidate = [reply[@"candidates"] firstObject];
	NSArray *parts = candidate[@"content"][@"parts"];
	XCTAssertEqualObjects(candidate[@"content"][@"role"], @"model");
	XCTAssertEqualObjects(parts[0], (@{ @"text": @"I should look.", @"thought": @YES }));
	XCTAssertEqualObjects(parts[1], @{ @"text": @"Checking." });
	XCTAssertEqualObjects(parts[2][@"functionCall"], (@{ @"id": @"c1", @"name": @"get_weather", @"args": @{ @"city": @"Paris" } }));
	XCTAssertEqualObjects(candidate[@"finishReason"], @"STOP", @"Gemini reports STOP for a call");
	XCTAssertEqualObjects(reply[@"usageMetadata"], (@{ @"promptTokenCount": @12, @"candidatesTokenCount": @5,
													   @"thoughtsTokenCount": @4, @"totalTokenCount": @21 }));
	XCTAssertEqualObjects(reply[@"modelVersion"], @"served");
}

- (void)testAReplyCutAtItsLimitFinishesAtMaxTokens
{
	NFKGeminiRouteBackend *backend = [[NFKGeminiRouteBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"Once upon", NFKOutputFinishReason: NFKFinishReasonLength }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend } configuration:nil];
	NSInteger status = 0;
	NSDictionary *reply = [self post:@"/v1beta/models/served:generateContent"
								body:@{ @"contents": @[ @{ @"parts": @[ @{ @"text": @"A story." } ] } ] } server:server status:&status];
	XCTAssertEqualObjects([reply[@"candidates"] firstObject][@"finishReason"], @"MAX_TOKENS");
	XCTAssertNil([[reply[@"candidates"] firstObject][@"content"][@"parts"] firstObject][@"thought"],
				 @"without includeThoughts no thought part is sent");
}

- (void)testAStreamSendsEachAdditionThenTheCallsAndTheFinish
{
	NFKGeminiRouteStreamingBackend *backend = [[NFKGeminiRouteStreamingBackend alloc] init];
	backend.partials = @[ [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"Hel" }],
						  [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"Hello" }] ];
	backend.finalResult = [NFKInferenceResult resultWithOutputs:@{
		NFKOutputText: @"Hello.", NFKOutputToolCalls: @[ @{ @"id": @"c1", @"name": @"lookup", @"arguments": @{} } ],
		NFKOutputUsage: @{ NFKUsageInputTokens: @3, NFKUsageOutputTokens: @2 } }];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend } configuration:nil];
	NSInteger status = 0;
	NSString *stream = [self send:@"POST" path:@"/v1beta/models/served:streamGenerateContent?alt=sse"
							 body:@{ @"contents": @[ @{ @"parts": @[ @{ @"text": @"Hi" } ] } ] } headers:nil server:server status:&status];
	XCTAssertEqual(status, 200);
	NSArray<NSDictionary *> *events = [self eventsIn:stream];
	NSMutableString *text = [NSMutableString string];
	for (NSDictionary *event in events) {
		for (NSDictionary *part in [event[@"candidates"] firstObject][@"content"][@"parts"]) {
			if ([part[@"text"] isKindOfClass:NSString.class]) {
				[text appendString:part[@"text"]];
			}
		}
	}
	XCTAssertEqualObjects(text, @"Hello.", @"each event carries only what the reply added");
	NSDictionary *last = [events.lastObject[@"candidates"] firstObject];
	XCTAssertEqualObjects(last[@"finishReason"], @"STOP");
	XCTAssertEqualObjects([last[@"content"][@"parts"] lastObject][@"functionCall"][@"name"], @"lookup");
	XCTAssertEqualObjects(events.lastObject[@"usageMetadata"][@"totalTokenCount"], @5);
}

#pragma mark Tokens, models, keys, and errors

- (void)testCountTokensTakesContentsOrAWholeRequest
{
	NFKGeminiRouteBackend *backend = [[NFKGeminiRouteBackend alloc] init];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend } configuration:nil];
	NSInteger status = 0;
	NSDictionary *reply = [self post:@"/v1beta/models/served:countTokens"
								body:@{ @"contents": @[ @{ @"parts": @[ @{ @"text": @"four" } ] } ] } server:server status:&status];
	XCTAssertEqualObjects(reply, @{ @"totalTokens": @4 });
	reply = [self post:@"/v1beta/models/served:countTokens"
				  body:@{ @"generateContentRequest": @{ @"model": @"models/served", @"contents": @[ @{ @"parts": @[ @{ @"text": @"seven!!" } ] } ] } }
				server:server status:&status];
	XCTAssertEqualObjects(reply, @{ @"totalTokens": @7 });
}

- (void)testTheModelListNamesTheHostedModels
{
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": [[NFKGeminiRouteBackend alloc] init] } configuration:nil];
	NSInteger status = 0;
	NSDictionary *list = [self JSONOf:[self send:@"GET" path:@"/v1beta/models" body:nil headers:nil server:server status:&status]];
	XCTAssertEqualObjects([list[@"models"] firstObject][@"name"], @"models/served");
	NSDictionary *entry = [self JSONOf:[self send:@"GET" path:@"/v1beta/models/served" body:nil headers:nil server:server status:&status]];
	XCTAssertTrue([entry[@"supportedGenerationMethods"] containsObject:@"generateContent"]);
	NSDictionary *missing = [self JSONOf:[self send:@"GET" path:@"/v1beta/models/absent" body:nil headers:nil server:server status:&status]];
	XCTAssertEqual(status, 404);
	XCTAssertEqualObjects(missing[@"error"][@"status"], @"NOT_FOUND");
	XCTAssertEqualObjects(missing[@"error"][@"code"], @404);
}

- (void)testAKeyIsTakenInGooglesForms
{
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": [[NFKGeminiRouteBackend alloc] init] }
											 configuration:^(NFKInferenceServer *configured) {
		configured.loopbackExemptScopes = 0;
		[configured addAPIKey:@"sesame" named:@"client" scopes:NFKServerKeyScopeInference];
	}];
	NSDictionary *body = @{ @"contents": @[ @{ @"parts": @[ @{ @"text": @"Hi" } ] } ] };
	NSInteger status = 0;
	NSDictionary *refused = [self JSONOf:[self send:@"POST" path:@"/v1beta/models/served:generateContent" body:body headers:nil
											 server:server status:&status]];
	XCTAssertEqual(status, 401);
	XCTAssertEqualObjects(refused[@"error"][@"status"], @"UNAUTHENTICATED");
	XCTAssertTrue([refused[@"error"][@"message"] containsString:@"x-goog-api-key"]);
	[self send:@"POST" path:@"/v1beta/models/served:generateContent" body:body headers:@{ @"x-goog-api-key": @"sesame" }
		server:server status:&status];
	XCTAssertEqual(status, 200);
	[self send:@"POST" path:@"/v1beta/models/served:generateContent?key=sesame" body:body headers:nil server:server status:&status];
	XCTAssertEqual(status, 200);
	[self send:@"POST" path:@"/v1/chat/completions?key=sesame" body:@{ @"model": @"served", @"messages": @[ @{ @"role": @"user", @"content": @"Hi" } ] }
	   headers:nil server:server status:&status];
	XCTAssertEqual(status, 401, @"the key query parameter opens Gemini's routes only");
}

- (void)testAnUnservedMethodAndAMalformedBodyAnswerInGooglesShape
{
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": [[NFKGeminiRouteBackend alloc] init] } configuration:nil];
	NSInteger status = 0;
	NSDictionary *unserved = [self post:@"/v1beta/models/served:embedContent" body:@{} server:server status:&status];
	XCTAssertEqual(status, 404);
	XCTAssertEqualObjects(unserved[@"error"][@"status"], @"NOT_FOUND");
	NSDictionary *empty = [self post:@"/v1beta/models/served:generateContent" body:@{ @"contents": @[] } server:server status:&status];
	XCTAssertEqual(status, 400);
	XCTAssertEqualObjects(empty[@"error"][@"status"], @"INVALID_ARGUMENT");
	NSDictionary *unanswered = [self post:@"/v1beta/models/served:generateContent"
									 body:@{ @"contents": @[ @{ @"parts": @[ @{ @"functionResponse": @{ @"name": @"lookup", @"response": @{} } } ] } ] }
								   server:server status:&status];
	XCTAssertEqual(status, 400, @"%@", unanswered);
}

#pragma mark NFKGeminiGenerateContentBackend against the server

- (void)testTheGeminiClientRunsAHostedModelAndItsToolTurns
{
	NFKGeminiRouteBackend *backend = [[NFKGeminiRouteBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"Bonjour.",
														NFKOutputUsage: @{ NFKUsageInputTokens: @11, NFKUsageOutputTokens: @2 } }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served": backend } configuration:^(NFKInferenceServer *configured) {
		configured.loopbackExemptScopes = 0;
		[configured addAPIKey:@"sesame" named:@"client" scopes:NFKServerKeyScopeInference];
	}];
	NFKGeminiGenerateContentBackend *client = [self clientFor:server model:@"served"];
	client.apiKey = @"sesame";
	NSArray *messages = @[
		@{ @"role": @"system", @"content": @"Answer in French." },
		@{ @"role": @"user", @"content": @"Weather?" },
		[NFKToolMessages assistantMessageWithText:nil toolCalls:@[ @{ @"id": @"c9", @"name": @"get_weather", @"arguments": @{ @"city": @"Paris" } } ]
										   record:nil format:nil source:nil],
		[NFKToolMessages messageForToolCallIdentifier:@"c9" name:@"get_weather"
											   result:[[NFKToolResult alloc] initWithText:@"18 C" outputs:nil imagePNGs:@[] isError:NO]],
	];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: messages }
															   parameters:@{ NFKParameterMaxTokens: @50 } outputModality:NFKModalityText];
	NSError *error = nil;
	NFKInferenceResult *result = [client runInferenceForRequest:request error:&error];
	XCTAssertEqualObjects(result.text, @"Bonjour.", @"%@", error);
	XCTAssertEqualObjects([result outputForKey:NFKOutputUsage][NFKUsageInputTokens], @11);
	XCTAssertEqualObjects([result outputForKey:NFKOutputFinishReason], NFKFinishReasonStop);
	NFKInferenceRequest *received = backend.lastRequest;
	XCTAssertEqualObjects(received.messages[0][@"content"], @"Answer in French.");
	XCTAssertEqualObjects([NFKToolMessages textOfToolMessage:received.messages[3]], @"18 C");
	XCTAssertEqualObjects([received parameterForKey:NFKParameterMaxTokens], @50);

	XCTestExpectation *ended = [self expectationWithDescription:@"the streamed run ends"];
	NFKInferenceJob *job = [client submitInferenceJobForRequest:request];
	job.completionHandler = ^(NFKInferenceJob *done) { [ended fulfill]; };
	[self waitForExpectations:@[ ended ] timeout:10];
	XCTAssertEqualObjects(job.result.text, @"Bonjour.", @"%@", job.error);
}

@end
