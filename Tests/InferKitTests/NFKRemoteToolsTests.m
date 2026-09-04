//
//  NFKRemoteToolsTests.m
//  InferKitTests
//
//  Tools, a schema, and several images through the blocking form of both chat backends: the wire
//  shape each provider takes, and what comes back under NFKOutputToolCalls and NFKOutputStructured.
//  A live tool call against a local runner is gated.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKAnthropicBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

@interface NFKToolsStubRemoteBackend : NFKRemoteBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, copy) NSString *stagedBody;
@end

@implementation NFKToolsStubRemoteBackend
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return [self.stagedBody dataUsingEncoding:NSUTF8StringEncoding];
}
@end

@interface NFKToolsStubAnthropicBackend : NFKAnthropicBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, copy) NSString *stagedBody;
@end

@implementation NFKToolsStubAnthropicBackend
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return [self.stagedBody dataUsingEncoding:NSUTF8StringEncoding];
}
@end

@interface NFKRemoteToolsTests : XCTestCase
@property (nonatomic, strong) NFKToolsStubRemoteBackend *backend;
@property (nonatomic, strong) NFKToolsStubAnthropicBackend *anthropic;
@property (nonatomic, copy) NSArray<NSDictionary *> *weatherTool;
@end

@implementation NFKRemoteToolsTests

- (void)setUp
{
	[super setUp];
	self.backend = [[NFKToolsStubRemoteBackend alloc] init];
	self.backend.endpointURL = [NSURL URLWithString:@"http://localhost:11434/v1/chat/completions"];
	self.backend.modelName = @"m";
	self.anthropic = [[NFKToolsStubAnthropicBackend alloc] init];
	self.anthropic.modelName = @"m";
	self.weatherTool = @[ @{ @"name": @"get_weather",
							 @"description": @"Current weather in a city.",
							 @"parameters": @{ @"type": @"object",
											   @"properties": @{ @"city": @{ @"type": @"string" } },
											   @"required": @[ @"city" ] } } ];
}

- (NSDictionary *)bodyOf:(NSURLRequest *)request
{
	return [NSJSONSerialization JSONObjectWithData:request.HTTPBody options:0 error:NULL];
}

- (NFKInferenceRequest *)prompt:(NSString *)text parameters:(NSDictionary *)parameters
{
	return [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: text } parameters:parameters outputModality:NFKModalityText];
}

- (CGImageRef)makeSquare CF_RETURNS_RETAINED
{
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 2, 2, 8, 8, colorSpace,
												 kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	CGContextSetRGBFillColor(context, 0, 0, 1, 1);
	CGContextFillRect(context, CGRectMake(0, 0, 2, 2));
	CGImageRef image = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	return image;
}

#pragma mark OpenAI-compatible

- (void)testToolsAreWrappedAsFunctionsAndACallComesBackParsed
{
	self.backend.stagedBody = @"{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":null,"
		"\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
		"\"arguments\":\"{\\\"city\\\": \\\"Paris\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}";
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[self prompt:@"weather in Paris?"
																		 parameters:@{ NFKParameterTools: self.weatherTool }]
																error:&error];
	XCTAssertNotNil(result, @"%@", error);

	NSDictionary *body = [self bodyOf:self.backend.lastRequest];
	XCTAssertEqualObjects(body[@"tools"][0][@"type"], @"function");
	XCTAssertEqualObjects(body[@"tools"][0][@"function"][@"name"], @"get_weather");
	XCTAssertEqualObjects(body[@"tools"][0][@"function"][@"parameters"][@"required"], @[ @"city" ]);
	XCTAssertNil(body[@"tools"][0][@"name"], @"the contract shape is translated, not sent as it was");
	XCTAssertEqual([body[@"tools"] count], 1);

	XCTAssertNil(result.text, @"a tool-calling turn has no text");
	XCTAssertEqual(result.toolCalls.count, 1);
	XCTAssertEqualObjects(result.toolCalls[0][@"id"], @"call_1");
	XCTAssertEqualObjects(result.toolCalls[0][@"name"], @"get_weather");
	XCTAssertEqualObjects(result.toolCalls[0][@"arguments"], (@{ @"city": @"Paris" }));
	XCTAssertEqualObjects(result.toolCalls[0][@"argumentsJSON"], @"{\"city\": \"Paris\"}");
}

// The contract key is spelled "tools", the wire field's own name, so a caller who already wrote the
// wire shape must find it sent as written rather than wrapped a second time.
- (void)testAToolAlreadyInTheWireShapePassesThroughUnwrapped
{
	self.backend.stagedBody = @"{\"choices\":[{\"message\":{\"content\":\"ok\"}}]}";
	NSDictionary *wire = @{ @"type": @"function", @"function": @{ @"name": @"get_weather", @"parameters": @{ @"type": @"object" } } };
	[self.backend runInferenceForRequest:[self prompt:@"x" parameters:@{ NFKParameterTools: @[ wire ] }] error:NULL];
	XCTAssertEqualObjects([self bodyOf:self.backend.lastRequest][@"tools"], @[ wire ]);

	self.anthropic.stagedBody = @"{\"content\":[{\"type\":\"text\",\"text\":\"ok\"}]}";
	NSDictionary *anthropicWire = @{ @"name": @"get_weather", @"input_schema": @{ @"type": @"object" } };
	[self.anthropic runInferenceForRequest:[self prompt:@"x" parameters:@{ NFKParameterTools: @[ anthropicWire ] }] error:NULL];
	XCTAssertEqualObjects([self bodyOf:self.anthropic.lastRequest][@"tools"], @[ anthropicWire ]);
}

- (void)testASchemaBecomesAJSONSchemaResponseFormatAndTheReplyIsParsed
{
	self.backend.stagedBody = @"{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"{\\\"answer\\\": 42, \\\"unit\\\": \\\"none\\\"}\"}}]}";
	NSDictionary *schema = @{ @"type": @"object", @"properties": @{ @"answer": @{ @"type": @"integer" } } };
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[self prompt:@"the answer?" parameters:@{ NFKParameterJSONSchema: schema }]
																error:NULL];
	NSDictionary *format = [self bodyOf:self.backend.lastRequest][@"response_format"];
	XCTAssertEqualObjects(format[@"type"], @"json_schema");
	XCTAssertEqualObjects(format[@"json_schema"][@"schema"], schema);
	XCTAssertEqualObjects(format[@"json_schema"][@"name"], @"response");

	XCTAssertEqualObjects(result.structured, (@{ @"answer": @42, @"unit": @"none" }));
	XCTAssertEqualObjects(result.text, @"{\"answer\": 42, \"unit\": \"none\"}", @"the text stays as sent");
}

- (void)testAResponseFormatFoldedInByNameIsAlsoParsedAndAnOrdinaryReplyIsNot
{
	self.backend.stagedBody = @"{\"choices\":[{\"message\":{\"content\":\"{\\\"ok\\\": true}\"}}]}";
	NFKInferenceResult *json = [self.backend runInferenceForRequest:[self prompt:@"x" parameters:@{ @"response_format": @{ @"type": @"json_object" } }]
															  error:NULL];
	XCTAssertEqualObjects(json.structured, (@{ @"ok": @YES }));

	NFKInferenceResult *plain = [self.backend runInferenceForRequest:[self prompt:@"x" parameters:@{}] error:NULL];
	XCTAssertNil(plain.structured, @"JSON-looking text is not promoted unless JSON was asked for");
	XCTAssertEqualObjects(plain.text, @"{\"ok\": true}");
}

- (void)testSeveralImagesAttachInOrderAfterTheFirst
{
	self.backend.stagedBody = @"{\"choices\":[{\"message\":{\"content\":\"same\"}}]}";
	CGImageRef a = [self makeSquare], b = [self makeSquare], c = [self makeSquare];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"compare",
																			  NFKInputImage: (__bridge id)a,
																			  NFKInputImages: @[ (__bridge id)b, (__bridge id)c ] }];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	NSArray *parts = [self bodyOf:self.backend.lastRequest][@"messages"][0][@"content"];
	XCTAssertEqual(parts.count, 4, @"the text and three images");
	XCTAssertEqualObjects(parts[0][@"type"], @"text");
	for (NSUInteger i = 1; i < 4; i++) {
		XCTAssertEqualObjects(parts[i][@"type"], @"image_url");
	}
	CGImageRelease(a); CGImageRelease(b); CGImageRelease(c);
}

#pragma mark Anthropic

- (void)testAnthropicTakesToolsInItsOwnShapeAndAToolUseComesBackAsACall
{
	self.anthropic.stagedBody = @"{\"content\":[{\"type\":\"text\",\"text\":\"Let me check.\"},"
		"{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{\"city\":\"Paris\"}}],\"stop_reason\":\"tool_use\"}";
	NFKInferenceResult *result = [self.anthropic runInferenceForRequest:[self prompt:@"weather?" parameters:@{ NFKParameterTools: self.weatherTool }]
																  error:NULL];
	NSDictionary *body = [self bodyOf:self.anthropic.lastRequest];
	XCTAssertEqualObjects(body[@"tools"][0][@"name"], @"get_weather");
	XCTAssertEqualObjects(body[@"tools"][0][@"input_schema"][@"required"], @[ @"city" ], @"parameters becomes input_schema");
	XCTAssertNil(body[@"tool_choice"], @"the model chooses");

	XCTAssertEqualObjects(result.text, @"Let me check.");
	XCTAssertEqual(result.toolCalls.count, 1);
	XCTAssertEqualObjects(result.toolCalls[0][@"id"], @"toolu_1");
	XCTAssertEqualObjects(result.toolCalls[0][@"arguments"], (@{ @"city": @"Paris" }));
	XCTAssertEqualObjects(result.toolCalls[0][@"argumentsJSON"], @"{\"city\":\"Paris\"}");
}

// The API has no response format; the schema is asked for through a forced tool, whose input is
// the structured reply rather than a call to act on.
- (void)testAnthropicAsksForASchemaThroughAForcedTool
{
	self.anthropic.stagedBody = @"{\"content\":[{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"structured_output\",\"input\":{\"answer\":42}}]}";
	NSDictionary *schema = @{ @"type": @"object", @"properties": @{ @"answer": @{ @"type": @"integer" } } };
	NFKInferenceResult *result = [self.anthropic runInferenceForRequest:[self prompt:@"the answer?" parameters:@{ NFKParameterJSONSchema: schema }]
																  error:NULL];
	NSDictionary *body = [self bodyOf:self.anthropic.lastRequest];
	XCTAssertEqualObjects(body[@"tool_choice"], (@{ @"type": @"tool", @"name": @"structured_output" }));
	XCTAssertEqualObjects(body[@"tools"][0][@"input_schema"], schema);

	XCTAssertEqualObjects(result.structured, (@{ @"answer": @42 }));
	XCTAssertNil(result.toolCalls);
	XCTAssertNil(result.text);
}

- (void)testAnthropicAttachesSeveralImagesBeforeTheText
{
	self.anthropic.stagedBody = @"{\"content\":[{\"type\":\"text\",\"text\":\"same\"}]}";
	CGImageRef a = [self makeSquare], b = [self makeSquare];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"compare",
																			  NFKInputImage: (__bridge id)a,
																			  NFKInputImages: @[ (__bridge id)b ] }];
	XCTAssertNotNil([self.anthropic runInferenceForRequest:request error:NULL]);
	NSArray *blocks = [self bodyOf:self.anthropic.lastRequest][@"messages"][0][@"content"];
	XCTAssertEqual(blocks.count, 3);
	XCTAssertEqualObjects(blocks[0][@"type"], @"image");
	XCTAssertEqualObjects(blocks[1][@"type"], @"image");
	XCTAssertEqualObjects(blocks[2][@"type"], @"text");
	CGImageRelease(a); CGImageRelease(b);
}

- (void)testTheResultAccessorIsNilForAnAbsentOrMistypedToolCallList
{
	XCTAssertNil([NFKInferenceResult resultWithOutputs:@{}].toolCalls);
	XCTAssertNil([NFKInferenceResult resultWithOutputs:@{ NFKOutputToolCalls: @"nope" }].toolCalls);
	XCTAssertEqual([NFKInferenceResult resultWithOutputs:@{ NFKOutputToolCalls: @[ @{ @"name": @"f" } ] }].toolCalls.count, 1);
}

#pragma mark A live local runner

// Set INFERKIT_LIVE_TOOL_MODEL to a tool-capable model the local server has (qwen3.5, gpt-oss). The
// question can only be answered by the tool, so the model has to call it with the city.
- (void)testALocalModelCallsTheTool
{
	NSDictionary *environment = NSProcessInfo.processInfo.environment;
	NSString *model = environment[@"INFERKIT_LIVE_TOOL_MODEL"];
	if (model.length == 0) {
		XCTSkip("set INFERKIT_LIVE_TOOL_MODEL (and run a local server) to exercise the live path");
	}
	NSString *identifier = environment[@"INFERKIT_LIVE_LOCAL_PROVIDER"] ?: @"ollama";
	id<NFKInferenceBackend> backend = [NFKRemoteProvider backendForProvider:[NFKRemoteProvider providerWithIdentifier:identifier]
																	 apiKey:nil modelName:model];
	NFKInferenceRequest *request = [self prompt:@"What is the weather in Paris right now? Use the tool."
									 parameters:@{ NFKParameterTools: self.weatherTool, NFKParameterMaxTokens: @1024 }];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error.localizedDescription);
	XCTAssertGreaterThan(result.toolCalls.count, 0, @"the model answered with text instead: %@", result.text);
	XCTAssertEqualObjects(result.toolCalls.firstObject[@"name"], @"get_weather");
	NSString *city = [result.toolCalls.firstObject[@"arguments"][@"city"] description];
	XCTAssertTrue([city.lowercaseString containsString:@"paris"], @"the model called with %@", result.toolCalls);
}

@end
