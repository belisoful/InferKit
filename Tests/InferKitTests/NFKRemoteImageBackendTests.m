//
//  NFKRemoteImageBackendTests.m
//  InferKitTests
//
//  The two image operations through a stub transport: a JSON generation, a multipart edit with and
//  without a mask, the two reply shapes (inline base64 and a URL fetched through the same seam), and
//  the provider factory's derivation.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteImageBackend.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>
#import <InferKit/NFKInferenceJob.h>

@interface NFKStubImageBackend : NFKRemoteImageBackend
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, copy) NSDictionary<NSString *, NSData *> *bodiesByURL;
@property (nonatomic, assign) NSInteger stagedStatusCode;
@property (nonatomic, copy, nullable) NSArray<NSString *> *stagedLines;
@end

@implementation NFKStubImageBackend

- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	if (self.requests == nil) {
		self.requests = [NSMutableArray array];
	}
	[self.requests addObject:request];
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:self.stagedStatusCode ?: 200
												  HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return self.bodiesByURL[request.URL.absoluteString] ?: [@"{}" dataUsingEncoding:NSUTF8StringEncoding];
}

- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse *, NSData *, NSError *))completionHandler
{
	if (self.requests == nil) {
		self.requests = [NSMutableArray array];
	}
	[self.requests addObject:request];
	for (NSString *line in self.stagedLines) {
		lineHandler(line);
	}
	completionHandler([[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil], nil, nil);
	return ^{};
}

@end

@interface NFKRemoteImageBackendTests : XCTestCase
@property (nonatomic, strong) NFKStubImageBackend *backend;
@property (nonatomic, assign) CGImageRef square;
@property (nonatomic, copy) NSData *squarePNG;
@end

@implementation NFKRemoteImageBackendTests

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
	self.squarePNG = [NFKImageCoding PNGDataForImage:(__bridge id)self.square];

	self.backend = [NFKStubImageBackend backendWithGenerationsURL:[NSURL URLWithString:@"https://api.openai.com/v1/images/generations"]
														 editsURL:[NSURL URLWithString:@"https://api.openai.com/v1/images/edits"]];
	self.backend.modelName = @"gpt-image-1";
	NSString *envelope = [NSString stringWithFormat:@"{\"created\":1,\"data\":[{\"b64_json\":\"%@\"}]}",
						[self.squarePNG base64EncodedStringWithOptions:0]];
	self.backend.bodiesByURL = @{
		@"https://api.openai.com/v1/images/generations": [envelope dataUsingEncoding:NSUTF8StringEncoding],
		@"https://api.openai.com/v1/images/edits": [envelope dataUsingEncoding:NSUTF8StringEncoding],
	};
}

- (void)tearDown
{
	CGImageRelease(self.square);
	[super tearDown];
}

- (NSDictionary *)decodedBodyOfRequest:(NSURLRequest *)request
{
	return [NSJSONSerialization JSONObjectWithData:request.HTTPBody options:0 error:NULL];
}

- (NSString *)multipartBodyOfRequest:(NSURLRequest *)request
{
	return [[NSString alloc] initWithData:request.HTTPBody encoding:NSISOLatin1StringEncoding];
}

- (void)testTheBackendReportsItsIdentity
{
	XCTAssertEqualObjects(self.backend.backendIdentifier, @"remote-image");
	XCTAssertTrue(self.backend.isReady);
	XCTAssertFalse([NFKRemoteImageBackend backendWithGenerationsURL:nil editsURL:nil].isReady);
}

#pragma mark Text to image

- (void)testAPromptAloneIsAJSONGenerationAndTheInlineImageDecodes
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a blue square" }
															  parameters:@{ NFKParameterWidth: @1024, NFKParameterHeight: @768,
																			NFKParameterSeed: @7, @"quality": @"high" }
														  outputModality:NFKModalityImage];
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);

	NSURLRequest *sent = self.backend.requests.firstObject;
	XCTAssertEqualObjects(sent.URL.absoluteString, @"https://api.openai.com/v1/images/generations");
	XCTAssertEqualObjects([sent valueForHTTPHeaderField:@"Content-Type"], @"application/json");
	NSDictionary *body = [self decodedBodyOfRequest:sent];
	XCTAssertEqualObjects(body[@"prompt"], @"a blue square");
	XCTAssertEqualObjects(body[@"model"], @"gpt-image-1");
	XCTAssertEqualObjects(body[@"size"], @"1024x768", @"width and height become the service's size");
	XCTAssertEqualObjects(body[@"seed"], @7);
	XCTAssertEqualObjects(body[@"quality"], @"high", @"an unmapped parameter passes by name");
	XCTAssertNil(body[NFKParameterWidth], @"the mapped keys do not also pass through");

	CVPixelBufferRef image = (__bridge CVPixelBufferRef)[result outputForKey:NFKOutputImage];
	XCTAssertTrue(image != NULL);
	XCTAssertEqual(CVPixelBufferGetWidth(image), 8);
	XCTAssertEqual(CVPixelBufferGetPixelFormatType(image), kCVPixelFormatType_32BGRA);
	XCTAssertNotNil([result outputForKey:NFKRemoteBackendRawKey]);
}

- (void)testAURLReplyIsFetchedThroughTheSameSeam
{
	self.backend.bodiesByURL = @{
		@"https://api.openai.com/v1/images/generations": [@"{\"data\":[{\"url\":\"https://cdn.example/out.png\"}]}" dataUsingEncoding:NSUTF8StringEncoding],
		@"https://cdn.example/out.png": self.squarePNG,
	};
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }]
																error:&error];
	XCTAssertNotNil(result, @"%@", error);
	XCTAssertEqual(self.backend.requests.count, 2);
	XCTAssertEqualObjects(self.backend.requests[1].URL.absoluteString, @"https://cdn.example/out.png");
	XCTAssertTrue([result outputForKey:NFKOutputImage] != nil);
}

#pragma mark Image to image

- (void)testAnImageMakesItAMultipartEditCarryingTheImageAsPNG
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"make it red",
																			  NFKInputImage: (__bridge id)self.square }];
	NSError *error = nil;
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:&error], @"%@", error);

	NSURLRequest *sent = self.backend.requests.firstObject;
	XCTAssertEqualObjects(sent.URL.absoluteString, @"https://api.openai.com/v1/images/edits");
	XCTAssertTrue([[sent valueForHTTPHeaderField:@"Content-Type"] hasPrefix:@"multipart/form-data; boundary="]);
	NSString *body = [self multipartBodyOfRequest:sent];
	XCTAssertTrue([body containsString:@"name=\"prompt\"\r\n\r\nmake it red"]);
	XCTAssertTrue([body containsString:@"name=\"model\"\r\n\r\ngpt-image-1"]);
	XCTAssertTrue([body containsString:@"name=\"image\"; filename=\"image.png\"\r\nContent-Type: image/png"]);
	XCTAssertFalse([body containsString:@"name=\"mask\""]);
	NSRange png = [sent.HTTPBody rangeOfData:self.squarePNG options:0 range:NSMakeRange(0, sent.HTTPBody.length)];
	XCTAssertNotEqual(png.location, NSNotFound, @"the PNG bytes are in the body verbatim");
}

- (void)testAMaskRidesAlongAsASecondFile
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"fill the hole",
																			  NFKInputImage: (__bridge id)self.square,
																			  NFKInputMask: (__bridge id)self.square }];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	NSString *body = [self multipartBodyOfRequest:self.backend.requests.firstObject];
	XCTAssertTrue([body containsString:@"name=\"mask\"; filename=\"mask.png\""]);
}

- (void)testAnEditWithoutAnEditEndpointIsUnsupportedAndAnUnencodableImageIsMissingInput
{
	self.backend.editsURL = nil;
	NSError *error = nil;
	NFKInferenceRequest *edit = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x",
																		   NFKInputImage: (__bridge id)self.square }];
	XCTAssertNil([self.backend runInferenceForRequest:edit error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);

	self.backend.editsURL = [NSURL URLWithString:@"https://api.openai.com/v1/images/edits"];
	NFKInferenceRequest *bogus = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x",
																			NFKInputImage: @"not an image" }];
	XCTAssertNil([self.backend runInferenceForRequest:bogus error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);
	XCTAssertNil(self.backend.requests, @"nothing was sent");
}

#pragma mark Failures

- (void)testAFailingStatusAndAnEmptyEnvelopeAreErrors
{
	NSError *error = nil;
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);

	self.backend.stagedStatusCode = 400;
	self.backend.bodiesByURL = @{ @"https://api.openai.com/v1/images/generations":
									  [@"{\"error\":{\"message\":\"size not supported\"}}" dataUsingEncoding:NSUTF8StringEncoding] };
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceRefused, @"a 400 is the request's problem");
	XCTAssertTrue([error.localizedDescription containsString:@"size not supported"]);

	self.backend.stagedStatusCode = 200;
	self.backend.bodiesByURL = @{ @"https://api.openai.com/v1/images/generations": [@"{\"data\":[]}" dataUsingEncoding:NSUTF8StringEncoding] };
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
}

#pragma mark The provider factory

#pragma mark Provider styles

- (NFKStubImageBackend *)stubFor:(NFKRemoteProvider *)provider model:(NSString *)model entries:(NSUInteger)entries
{
	NFKRemoteImageBackend *made = [NFKRemoteImageBackend backendForProvider:provider apiKey:@"k" modelName:model];
	NFKStubImageBackend *stub = [NFKStubImageBackend backendWithGenerationsURL:made.generationsURL editsURL:made.editsURL];
	stub.apiStyle = made.apiStyle;
	stub.modelName = model;
	NSMutableArray<NSString *> *data = [NSMutableArray array];
	for (NSUInteger index = 0; index < entries; index++) {
		[data addObject:[NSString stringWithFormat:@"{\"b64_json\":\"%@\"}", [self.squarePNG base64EncodedStringWithOptions:0]]];
	}
	NSData *envelope = [[NSString stringWithFormat:@"{\"data\":[%@]}", [data componentsJoinedByString:@","]] dataUsingEncoding:NSUTF8StringEncoding];
	NSMutableDictionary *bodies = [NSMutableDictionary dictionary];
	if (made.generationsURL != nil) {
		bodies[made.generationsURL.absoluteString] = envelope;
	}
	if (made.editsURL != nil) {
		bodies[made.editsURL.absoluteString] = envelope;
	}
	stub.bodiesByURL = bodies;
	return stub;
}

- (void)testEachProviderGetsItsOwnPathsAndTheImagelessOnesNone
{
	NFKRemoteImageBackend *router = [NFKRemoteImageBackend backendForProvider:NFKRemoteProvider.openRouter apiKey:@"k" modelName:@"m"];
	XCTAssertEqualObjects(router.generationsURL.absoluteString, @"https://openrouter.ai/api/v1/images");
	NFKRemoteImageBackend *together = [NFKRemoteImageBackend backendForProvider:NFKRemoteProvider.together apiKey:@"k" modelName:@"m"];
	XCTAssertEqualObjects(together.editsURL, together.generationsURL, @"Together edits on its generations path");
	XCTAssertNil([NFKRemoteImageBackend backendForProvider:NFKRemoteProvider.googleGemini apiKey:@"k" modelName:@"m"].editsURL);
	for (NFKRemoteProvider *provider in @[ NFKRemoteProvider.groq, NFKRemoteProvider.mistral, NFKRemoteProvider.deepSeek, NFKRemoteProvider.ollama ]) {
		XCTAssertNil([NFKRemoteImageBackend backendForProvider:provider apiKey:@"k" modelName:@"m"], @"%@", provider.identifier);
	}
}

// xAI edits by JSON: several sources as images[] of data-URI objects, a size as a ratio.
- (void)testXAIEditsSeveralImagesByJSONAndDerivesTheRatio
{
	NFKStubImageBackend *xai = [self stubFor:NFKRemoteProvider.xAI model:@"grok-imagine-image-2.0" entries:2];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"merge", NFKInputImage: (__bridge id)self.square,
																			  NFKInputImages: @[ (__bridge id)self.square ] }
															   parameters:@{ NFKParameterWidth: @1920, NFKParameterHeight: @1080, NFKParameterSampleCount: @2 }];
	NSError *error = nil;
	NFKInferenceResult *result = [xai runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);
	NSURLRequest *sent = xai.requests.firstObject;
	XCTAssertEqualObjects(sent.URL.absoluteString, @"https://api.x.ai/v1/images/edits");
	XCTAssertEqualObjects([sent valueForHTTPHeaderField:@"Content-Type"], @"application/json");
	NSDictionary *body = [self decodedBodyOfRequest:sent];
	XCTAssertEqual([body[@"images"] count], 2);
	XCTAssertTrue([body[@"images"][0][@"url"] hasPrefix:@"data:image/png;base64,"]);
	XCTAssertEqualObjects(body[@"aspect_ratio"], @"16:9");
	XCTAssertEqualObjects(body[@"n"], @2);
	XCTAssertEqualObjects(body[@"response_format"], @"b64_json");
	XCTAssertEqual([[result outputForKey:NFKOutputImages] count], 2);

	NFKInferenceRequest *masked = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x", NFKInputImage: (__bridge id)self.square,
																			 NFKInputMask: (__bridge id)self.square }];
	XCTAssertNil([xai runInferenceForRequest:masked error:&error]);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
}

// Together generates and edits on one path, with width and height and the source as image_url.
- (void)testTogetherSendsWidthHeightAndTheSourceOnItsGenerationsPath
{
	NFKStubImageBackend *together = [self stubFor:NFKRemoteProvider.together model:@"black-forest-labs/FLUX.1-kontext-pro" entries:1];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"night", NFKInputImage: (__bridge id)self.square,
																			  NFKInputNegativePrompt: @"blur" }
															   parameters:@{ NFKParameterWidth: @1024, NFKParameterHeight: @768, NFKParameterSteps: @28,
																			 NFKParameterSeed: @3, NFKParameterGuidanceScale: @3.5 }];
	XCTAssertNotNil([together runInferenceForRequest:request error:NULL]);
	NSURLRequest *sent = together.requests.firstObject;
	XCTAssertEqualObjects(sent.URL.absoluteString, @"https://api.together.xyz/v1/images/generations");
	NSDictionary *body = [self decodedBodyOfRequest:sent];
	XCTAssertEqualObjects(body[@"width"], @1024);
	XCTAssertEqualObjects(body[@"height"], @768);
	XCTAssertNil(body[@"size"]);
	XCTAssertEqualObjects(body[@"steps"], @28);
	XCTAssertEqualObjects(body[@"guidance_scale"], @3.5);
	XCTAssertEqualObjects(body[@"negative_prompt"], @"blur");
	XCTAssertEqualObjects(body[@"response_format"], @"base64");
	XCTAssertTrue([body[@"image_url"] hasPrefix:@"data:image/png;base64,"]);
}

- (void)testOpenRouterPostsToImagesWithInputReferences
{
	NFKStubImageBackend *router = [self stubFor:NFKRemoteProvider.openRouter model:@"google/gemini-3.1-flash-image" entries:1];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a fox", NFKInputImage: (__bridge id)self.square }
															   parameters:@{ NFKParameterAspectRatio: @"1:1", NFKParameterResolution: @"2K", @"quality": @"high" }];
	XCTAssertNotNil([router runInferenceForRequest:request error:NULL]);
	NSDictionary *body = [self decodedBodyOfRequest:router.requests.firstObject];
	XCTAssertEqualObjects(router.requests.firstObject.URL.absoluteString, @"https://openrouter.ai/api/v1/images");
	XCTAssertEqual([body[@"input_references"] count], 1);
	XCTAssertEqualObjects(body[@"resolution"], @"2K");
	XCTAssertEqualObjects(body[@"quality"], @"high");
}

- (void)testOpenAIEditsSeveralImagesAsRepeatedImageFields
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"collage", NFKInputImage: (__bridge id)self.square,
																			  NFKInputImages: @[ (__bridge id)self.square, (__bridge id)self.square ] }];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	NSString *body = [self multipartBodyOfRequest:self.backend.requests.firstObject];
	XCTAssertEqual([body componentsSeparatedByString:@"name=\"image[]\""].count, 4, @"three sources, each an image[] file");
}

- (void)testAStreamedGenerationReportsPartialImagesAndFinishesWithTheCompletedOne
{
	self.backend.streams = YES;
	NSString *encoded = [self.squarePNG base64EncodedStringWithOptions:0];
	self.backend.stagedLines = @[ [NSString stringWithFormat:@"data: {\"type\":\"image_generation.partial_image\",\"b64_json\":\"%@\",\"partial_image_index\":0}", encoded],
								  [NSString stringWithFormat:@"data: {\"type\":\"image_generation.completed\",\"b64_json\":\"%@\"}", encoded] ];
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a square" }]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	XCTAssertNotNil([job.partialResult outputForKey:NFKOutputImage], @"the partial image arrived before the completed one");
	XCTAssertNotNil([job.result outputForKey:NFKOutputImage]);
	NSDictionary *body = [self decodedBodyOfRequest:self.backend.requests.firstObject];
	XCTAssertEqualObjects(body[@"stream"], @YES);
	XCTAssertEqualObjects(body[@"partial_images"], @2);
}

- (void)testTheFactoryDerivesBothURLsAndDeclinesAnthropic
{
	NFKRemoteImageBackend *xai = [NFKRemoteImageBackend backendForProvider:NFKRemoteProvider.xAI apiKey:@"k" modelName:@"grok-2-image"];
	XCTAssertEqualObjects(xai.generationsURL.absoluteString, @"https://api.x.ai/v1/images/generations");
	XCTAssertEqualObjects(xai.editsURL.absoluteString, @"https://api.x.ai/v1/images/edits");
	XCTAssertEqualObjects(xai.apiKey, @"k");
	XCTAssertNil([NFKRemoteImageBackend backendForProvider:NFKRemoteProvider.anthropic apiKey:@"k" modelName:@"m"]);
}

@end
