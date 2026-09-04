//
//  NFKRemoteModerationBackendTests.m
//  InferKitTests
//
//  The moderation request and verdict through a stub transport: text alone, text with an image,
//  the per-category classifications, and the provider factory.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteModerationBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKClassification.h>
#import <InferKit/NFKErrors.h>

@interface NFKStubModerationBackend : NFKRemoteModerationBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, copy) NSString *stagedBody;
@end

@implementation NFKStubModerationBackend
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return [self.stagedBody dataUsingEncoding:NSUTF8StringEncoding];
}
@end

@interface NFKRemoteModerationBackendTests : XCTestCase
@property (nonatomic, strong) NFKStubModerationBackend *backend;
@end

@implementation NFKRemoteModerationBackendTests

- (void)setUp
{
	[super setUp];
	self.backend = [[NFKStubModerationBackend alloc] init];
	self.backend.endpointURL = [NSURL URLWithString:@"https://api.openai.com/v1/moderations"];
	self.backend.modelName = @"omni-moderation-latest";
	self.backend.stagedBody = @"{\"results\":[{\"flagged\":true,"
		"\"categories\":{\"harassment\":true,\"violence\":false,\"self-harm\":false},"
		"\"category_scores\":{\"harassment\":0.91,\"violence\":0.12,\"self-harm\":0.01}}]}";
}

- (NSDictionary *)decodedRequestBody
{
	return [NSJSONSerialization JSONObjectWithData:self.backend.lastRequest.HTTPBody options:0 error:NULL];
}

- (void)testTextIsSentAsTheInputAndTheVerdictComesBackAsClassifications
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"some text" }];
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);

	NSDictionary *body = [self decodedRequestBody];
	XCTAssertEqualObjects(body[@"input"], @"some text");
	XCTAssertEqualObjects(body[@"model"], @"omni-moderation-latest");

	NSArray<NFKClassification *> *classifications = result.classifications;
	XCTAssertEqual(classifications.count, 3);
	XCTAssertEqualObjects(classifications[0].label, @"harassment", @"most confident first");
	XCTAssertEqualWithAccuracy(classifications[0].confidence, 0.91, 1e-9);
	XCTAssertEqualObjects(classifications[2].label, @"self-harm");
	XCTAssertEqualObjects(result.structured[@"flagged"], @YES);
	XCTAssertEqualObjects(result.structured[@"categories"][@"harassment"], @YES);
	XCTAssertEqualObjects(self.backend.backendIdentifier, @"remote-moderation");
}

- (void)testAnImageMakesTheInputTheMultimodalPartsList
{
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 2, 2, 8, 8, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	CGImageRef square = CGBitmapContextCreateImage(context);
	CGContextRelease(context);

	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"caption", NFKInputImage: (__bridge id)square }];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	NSArray *input = [self decodedRequestBody][@"input"];
	XCTAssertEqual(input.count, 2);
	XCTAssertEqualObjects(input[0], (@{ @"type": @"text", @"text": @"caption" }));
	XCTAssertEqualObjects(input[1][@"type"], @"image_url");
	CGImageRelease(square);
}

- (void)testAnEmptyRequestAndAVerdictlessReplyAreErrors
{
	NSError *error = nil;
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);

	self.backend.stagedBody = @"{\"results\":[]}";
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
}

- (void)testTheFactoryDerivesTheModerationsURLAndDeclinesAnthropic
{
	NFKRemoteModerationBackend *mistral = [NFKRemoteModerationBackend backendForProvider:NFKRemoteProvider.mistral
																				 apiKey:@"k" modelName:@"mistral-moderation-latest"];
	XCTAssertEqualObjects(mistral.endpointURL.absoluteString, @"https://api.mistral.ai/v1/moderations");
	XCTAssertNil([NFKRemoteModerationBackend backendForProvider:NFKRemoteProvider.anthropic apiKey:@"k" modelName:@"m"]);
}

@end
