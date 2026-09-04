//
//  NFKRemoteVisionTests.m
//  InferKitTests
//
//  An image beside the prompt: the content-part shape the OpenAI-compatible chat endpoint reads and
//  the block shape Anthropic reads, through stub transports. A live pass against a local vision
//  model is gated on one being named.
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

@interface NFKVisionStubRemoteBackend : NFKRemoteBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@end

@implementation NFKVisionStubRemoteBackend
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return [@"{\"choices\":[{\"message\":{\"content\":\"a blue square\"}}]}" dataUsingEncoding:NSUTF8StringEncoding];
}
@end

@interface NFKVisionStubAnthropicBackend : NFKAnthropicBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@end

@implementation NFKVisionStubAnthropicBackend
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return [@"{\"content\":[{\"type\":\"text\",\"text\":\"a blue square\"}]}" dataUsingEncoding:NSUTF8StringEncoding];
}
@end

@interface NFKRemoteVisionTests : XCTestCase
@property (nonatomic, assign) CGImageRef square;
@property (nonatomic, copy) NSString *squareBase64;
@end

@implementation NFKRemoteVisionTests

- (void)setUp
{
	[super setUp];
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 8, 8, 8, 32, colorSpace,
												 kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	CGContextSetRGBFillColor(context, 0, 0, 1, 1);
	CGContextFillRect(context, CGRectMake(0, 0, 8, 8));
	self.square = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	self.squareBase64 = [[NFKImageCoding PNGDataForImage:(__bridge id)self.square] base64EncodedStringWithOptions:0];
}

- (void)tearDown
{
	CGImageRelease(self.square);
	[super tearDown];
}

- (NSDictionary *)bodyOf:(NSURLRequest *)request
{
	return [NSJSONSerialization JSONObjectWithData:request.HTTPBody options:0 error:NULL];
}

#pragma mark OpenAI-compatible chat

- (void)testAnImageBesideThePromptBecomesContentPartsOnTheUserTurn
{
	NFKVisionStubRemoteBackend *backend = [[NFKVisionStubRemoteBackend alloc] init];
	backend.endpointURL = [NSURL URLWithString:@"http://localhost:11434/v1/chat/completions"];
	backend.modelName = @"qwen3.5:27b";
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"What color is this?",
																			  NFKInputImage: (__bridge id)self.square }];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
	XCTAssertEqualObjects(result.text, @"a blue square", @"%@", error);

	NSArray *messages = [self bodyOf:backend.lastRequest][@"messages"];
	XCTAssertEqual(messages.count, 1);
	NSArray *parts = messages[0][@"content"];
	XCTAssertTrue([parts isKindOfClass:NSArray.class], @"a string content became parts");
	XCTAssertEqualObjects(parts[0], (@{ @"type": @"text", @"text": @"What color is this?" }));
	XCTAssertEqualObjects(parts[1][@"type"], @"image_url");
	XCTAssertEqualObjects(parts[1][@"image_url"][@"url"], [@"data:image/png;base64," stringByAppendingString:self.squareBase64]);
}

- (void)testTheImageAttachesToTheLastUserTurnOfAConversation
{
	NFKVisionStubRemoteBackend *backend = [[NFKVisionStubRemoteBackend alloc] init];
	backend.endpointURL = [NSURL URLWithString:@"http://localhost:11434/v1/chat/completions"];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{
		NFKInputMessages: @[ @{ @"role": @"system", @"content": @"Be brief." },
							 @{ @"role": @"user", @"content": @"first" },
							 @{ @"role": @"assistant", @"content": @"ok" },
							 @{ @"role": @"user", @"content": @"and this?" } ],
		NFKInputImage: (__bridge id)self.square }];
	[backend runInferenceForRequest:request error:NULL];

	NSArray *messages = [self bodyOf:backend.lastRequest][@"messages"];
	XCTAssertEqual(messages.count, 4);
	XCTAssertEqualObjects(messages[1][@"content"], @"first", @"an earlier turn is untouched");
	NSArray *parts = messages[3][@"content"];
	XCTAssertEqual(parts.count, 2);
	XCTAssertEqualObjects(parts[0][@"text"], @"and this?");
	XCTAssertEqualObjects(parts[1][@"type"], @"image_url");
}

- (void)testAnUnencodableImageIsRefusedBeforeAnyRequest
{
	NFKVisionStubRemoteBackend *backend = [[NFKVisionStubRemoteBackend alloc] init];
	backend.endpointURL = [NSURL URLWithString:@"http://localhost:11434/v1/chat/completions"];
	NSError *error = nil;
	NFKInferenceRequest *bogus = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x", NFKInputImage: @"nope" }];
	XCTAssertNil([backend runInferenceForRequest:bogus error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);
	XCTAssertNil(backend.lastRequest);
}

#pragma mark Anthropic

- (void)testAnthropicTakesTheImageAsABase64BlockBeforeTheText
{
	NFKVisionStubAnthropicBackend *backend = [[NFKVisionStubAnthropicBackend alloc] init];
	backend.modelName = @"claude-sonnet-4-5";
	backend.apiKey = @"k";
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"What color is this?",
																			  NFKInputImage: (__bridge id)self.square }];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
	XCTAssertEqualObjects(result.text, @"a blue square", @"%@", error);

	NSArray *blocks = [self bodyOf:backend.lastRequest][@"messages"][0][@"content"];
	XCTAssertEqual(blocks.count, 2);
	XCTAssertEqualObjects(blocks[0][@"type"], @"image");
	XCTAssertEqualObjects(blocks[0][@"source"][@"type"], @"base64");
	XCTAssertEqualObjects(blocks[0][@"source"][@"media_type"], @"image/png");
	XCTAssertEqualObjects(blocks[0][@"source"][@"data"], self.squareBase64);
	XCTAssertEqualObjects(blocks[1], (@{ @"type": @"text", @"text": @"What color is this?" }));
}

#pragma mark A live local vision model

// Set INFERKIT_LIVE_VISION_MODEL to a vision-capable model the local server has (qwen3.5, llava,
// gemma4). The image is a flat blue square, so the answer has to name the color.
- (void)testALocalVisionModelSeesTheImage
{
	NSDictionary *environment = NSProcessInfo.processInfo.environment;
	NSString *model = environment[@"INFERKIT_LIVE_VISION_MODEL"];
	if (model.length == 0) {
		XCTSkip("set INFERKIT_LIVE_VISION_MODEL (and run a local server) to exercise the live path");
	}
	NSString *identifier = environment[@"INFERKIT_LIVE_LOCAL_PROVIDER"] ?: @"ollama";
	NFKRemoteProvider *provider = [NFKRemoteProvider providerWithIdentifier:identifier];
	id<NFKInferenceBackend> backend = [NFKRemoteProvider backendForProvider:provider apiKey:nil modelName:model];

	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 64, 64, 8, 256, colorSpace,
												 kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	CGContextSetRGBFillColor(context, 0, 0, 1, 1);
	CGContextFillRect(context, CGRectMake(0, 0, 64, 64));
	CGImageRef blue = CGBitmapContextCreateImage(context);
	CGContextRelease(context);

	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{
		NFKInputPrompt: @"What single color fills this image? Answer with one word.",
		NFKInputImage: (__bridge id)blue }
		parameters:@{ NFKParameterMaxTokens: @512 } outputModality:NFKModalityText];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
	CGImageRelease(blue);
	XCTAssertNotNil(result, @"%@", error.localizedDescription);
	NSString *answer = [result outputForKey:NFKRemoteBackendTextKey];
	XCTAssertTrue([answer.lowercaseString containsString:@"blue"], @"the model answered: %@", answer);
}

@end
