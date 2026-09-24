//
//  NFKRemoteVideoBackendTests.m
//  InferKitTests
//
//  The job-style video service through stubbed transports: the submit (JSON, or multipart with a
//  reference image), the polls and their percentage progress, the content download that a finished
//  job becomes, and the service's own failure message.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteVideoBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKVideoAsset.h>
#import <InferKit/NFKErrors.h>

@interface NFKStubVideoBackend : NFKRemoteVideoBackend
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *jsonRequests;
@property (nonatomic, strong) NSMutableArray<NSDictionary *> *stagedJSON;
@property (nonatomic, strong) NSURLRequest *downloadRequest;
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *downloadRequests;
@property (nonatomic, copy) NSData *stagedClip;
@end

@implementation NFKStubVideoBackend
- (NSDictionary *)sendJSONRequest:(NSURLRequest *)request error:(NSError **)outError
{
	if (self.jsonRequests == nil) {
		self.jsonRequests = [NSMutableArray array];
	}
	[self.jsonRequests addObject:request];
	NSDictionary *next = self.stagedJSON.firstObject;
	if (self.stagedJSON.count > 0) {
		[self.stagedJSON removeObjectAtIndex:0];
	}
	return next;
}
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.downloadRequest = request;
	if (self.downloadRequests == nil) {
		self.downloadRequests = [NSMutableArray array];
	}
	[self.downloadRequests addObject:request];
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return self.stagedClip;
}
@end

@interface NFKRemoteVideoBackendTests : XCTestCase
@property (nonatomic, strong) NFKStubVideoBackend *backend;
@property (nonatomic, strong) NSURL *directory;
@end

@implementation NFKRemoteVideoBackendTests

- (void)setUp
{
	[super setUp];
	self.directory = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:NSUUID.UUID.UUIDString isDirectory:YES];
	self.backend = [[NFKStubVideoBackend alloc] init];
	self.backend.submitURL = [NSURL URLWithString:@"https://api.openai.com/v1/videos"];
	self.backend.modelName = @"sora-2";
	self.backend.apiKey = @"k";
	self.backend.pollInterval = 0;
	self.backend.outputDirectoryURL = self.directory;
	self.backend.stagedClip = [@"ftypisom fake mp4" dataUsingEncoding:NSUTF8StringEncoding];
	self.backend.stagedJSON = [@[
		@{ @"id": @"video_1", @"object": @"video", @"status": @"queued", @"progress": @0 },
		@{ @"id": @"video_1", @"status": @"in_progress", @"progress": @40 },
		@{ @"id": @"video_1", @"status": @"completed", @"progress": @100 },
	] mutableCopy];
}

- (void)tearDown
{
	[NSFileManager.defaultManager removeItemAtURL:self.directory error:NULL];
	[super tearDown];
}

- (NSDictionary *)bodyOf:(NSURLRequest *)request
{
	return [NSJSONSerialization JSONObjectWithData:request.HTTPBody options:0 error:NULL];
}

- (void)testAPromptSubmitsPollsToCompletionAndDownloadsTheClip
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a lighthouse at dusk" }
															  parameters:@{ NFKParameterDurationSeconds: @8,
																			NFKParameterWidth: @1280, NFKParameterHeight: @720 }
														  outputModality:NFKModalityVideo];
	NSMutableArray<NSNumber *> *progress = [NSMutableArray array];
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:request];
	job.progressHandler = ^(NFKInferenceJob *j) { @synchronized (progress) { [progress addObject:@(j.progress)]; } };
	XCTestExpectation *ended = [self expectationWithDescription:@"job ended"];
	job.completionHandler = ^(NFKInferenceJob *j) { [ended fulfill]; };
	[self waitForExpectations:@[ ended ] timeout:10];

	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	NSURLRequest *submit = self.backend.jsonRequests.firstObject;
	XCTAssertEqualObjects(submit.URL.absoluteString, @"https://api.openai.com/v1/videos");
	XCTAssertEqualObjects(submit.HTTPMethod, @"POST");
	NSDictionary *body = [self bodyOf:submit];
	XCTAssertEqualObjects(body[@"model"], @"sora-2");
	XCTAssertEqualObjects(body[@"prompt"], @"a lighthouse at dusk");
	XCTAssertEqualObjects(body[@"seconds"], @"8", @"the service takes the length as a string");
	XCTAssertEqualObjects(body[@"size"], @"1280x720");
	XCTAssertEqualObjects(submit.allHTTPHeaderFields[@"Authorization"], @"Bearer k");

	XCTAssertEqual(self.backend.jsonRequests.count, 3, @"a submit and two polls; the completed poll ends it");
	XCTAssertEqualObjects(self.backend.jsonRequests[1].URL.absoluteString, @"https://api.openai.com/v1/videos/video_1");
	XCTAssertTrue([progress containsObject:@0.4], @"the percentage became a fraction: %@", progress);

	XCTAssertEqualObjects(self.backend.downloadRequest.URL.absoluteString, @"https://api.openai.com/v1/videos/video_1/content");
	NFKVideoAsset *clip = [job.result outputForKey:NFKOutputVideo];
	XCTAssertEqualObjects(clip.fileURL.pathExtension, @"mp4");
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:clip.fileURL], self.backend.stagedClip);
	XCTAssertTrue([clip.fileURL.path hasPrefix:self.directory.path]);
}

- (void)testAReferenceImageMakesTheSubmitMultipart
{
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 4, 4, 8, 16, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	CGImageRef square = CGBitmapContextCreateImage(context);
	CGContextRelease(context);

	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"animate this",
																			  NFKInputImage: (__bridge id)square }];
	NSError *error = nil;
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:&error], @"%@", error);
	NSURLRequest *submit = self.backend.jsonRequests.firstObject;
	XCTAssertTrue([[submit valueForHTTPHeaderField:@"Content-Type"] hasPrefix:@"multipart/form-data; boundary="]);
	NSString *body = [[NSString alloc] initWithData:submit.HTTPBody encoding:NSISOLatin1StringEncoding];
	XCTAssertTrue([body containsString:@"name=\"prompt\"\r\n\r\nanimate this"]);
	XCTAssertTrue([body containsString:@"name=\"input_reference\"; filename=\"reference.png\"\r\nContent-Type: image/png"]);
	CGImageRelease(square);
}

- (void)testAFailedJobCarriesTheServicesOwnReason
{
	self.backend.stagedJSON = [@[
		@{ @"id": @"video_2", @"status": @"queued" },
		@{ @"id": @"video_2", @"status": @"failed", @"error": @{ @"code": @"moderation_blocked", @"message": @"prompt rejected" } },
	] mutableCopy];
	NSError *error = nil;
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }] error:&error]);
	XCTAssertTrue([error.localizedDescription containsString:@"prompt rejected"], @"%@", error);
	XCTAssertNil(self.backend.downloadRequest, @"nothing was downloaded");
}

#pragma mark Provider styles

- (CGImageRef)newSquare CF_RETURNS_RETAINED
{
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 4, 4, 8, 16, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	CGImageRef square = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	return square;
}

- (NFKStubVideoBackend *)stubForProvider:(NFKRemoteProvider *)provider style:(NFKRemoteVideoAPIStyle)style model:(NSString *)model
{
	NFKRemoteVideoBackend *made = [NFKRemoteVideoBackend backendForProvider:provider apiStyle:style apiKey:@"k" modelName:model];
	NFKStubVideoBackend *stub = [[NFKStubVideoBackend alloc] init];
	stub.apiStyle = made.apiStyle;
	stub.submitURL = made.submitURL;
	stub.apiKey = @"k";
	stub.modelName = model;
	stub.pollInterval = 0;
	stub.outputDirectoryURL = self.directory;
	stub.stagedClip = [@"ftypisom fake mp4" dataUsingEncoding:NSUTF8StringEncoding];
	return stub;
}

- (NSString *)multipartTextOf:(NSURLRequest *)request
{
	return [[NSString alloc] initWithData:request.HTTPBody encoding:NSISOLatin1StringEncoding];
}

- (void)testEachProviderGetsItsOwnStyleAndPath
{
	NSDictionary<NSString *, NSString *> *expected = @{ @"openai": @"https://api.openai.com/v1/videos",
														@"gemini": @"https://generativelanguage.googleapis.com/v1beta/openai/videos",
														@"xai": @"https://api.x.ai/v1/videos/generations",
														@"together": @"https://api.together.xyz/v2/videos",
														@"openrouter": @"https://openrouter.ai/api/v1/videos" };
	for (NSString *identifier in expected) {
		NFKRemoteVideoBackend *backend = [NFKRemoteVideoBackend backendForProvider:[NFKRemoteProvider providerWithIdentifier:identifier]
																			apiKey:@"k" modelName:@"m"];
		XCTAssertEqualObjects(backend.submitURL.absoluteString, expected[identifier], @"%@", identifier);
	}
	XCTAssertEqual([NFKRemoteVideoBackend backendForProvider:NFKRemoteProvider.googleGemini apiKey:@"k" modelName:@"m"].apiStyle,
				   NFKRemoteVideoAPIStyleGeminiSoraCompatible);
	NFKRemoteVideoBackend *veo = [NFKRemoteVideoBackend backendForProvider:NFKRemoteProvider.googleGemini apiStyle:NFKRemoteVideoAPIStyleGeminiVeo
																	apiKey:@"k" modelName:@"veo-3.1-generate-preview"];
	XCTAssertEqualObjects(veo.submitURL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/models");
	XCTAssertNil([NFKRemoteVideoBackend backendForProvider:NFKRemoteProvider.groq apiKey:@"k" modelName:@"m"], @"Groq serves no video");
	XCTAssertNil([NFKRemoteVideoBackend backendForProvider:NFKRemoteProvider.openAI apiStyle:NFKRemoteVideoAPIStyleXAI apiKey:@"k" modelName:@"m"]);
}

// Gemini's Sora-compatible layer: a multipart submit whose Veo options are form fields.
- (void)testGeminiSoraCompatibleSendsVeoOptionsAsFormFieldsAndFetchesTheJobsURL
{
	NFKStubVideoBackend *gemini = [self stubForProvider:NFKRemoteProvider.googleGemini style:NFKRemoteVideoAPIStyleGeminiSoraCompatible
												  model:@"veo-3.1-generate-preview"];
	gemini.stagedJSON = [@[ @{ @"id": @"op1", @"status": @"processing" },
							@{ @"id": @"op1", @"status": @"completed",
							   @"url": @"https://generativelanguage.googleapis.com/v1beta/files/clip:download?alt=media" } ] mutableCopy];
	CGImageRef first = [self newSquare];
	CGImageRef reference = [self newSquare];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a waterfall",
																			  NFKInputNegativePrompt: @"shaky camera",
																			  NFKInputImage: (__bridge id)first,
																			  NFKInputImages: @[ (__bridge id)reference, (__bridge id)reference ] }
																parameters:@{ NFKParameterDurationSeconds: @8, NFKParameterAspectRatio: @"9:16",
																			  NFKParameterResolution: @"1080p", NFKParameterSeed: @7,
																			  NFKParameterFramesPerSecond: @24, @"person_generation": @"allow_adult" }
															outputModality:NFKModalityVideo];
	NSError *error = nil;
	NFKInferenceResult *result = [gemini runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);
	NSURLRequest *submit = gemini.jsonRequests.firstObject;
	XCTAssertEqualObjects(submit.URL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/openai/videos");
	XCTAssertEqualObjects(submit.allHTTPHeaderFields[@"Authorization"], @"Bearer k");
	NSString *form = [self multipartTextOf:submit];
	for (NSString *field in @[ @"name=\"model\"\r\n\r\nveo-3.1-generate-preview", @"name=\"duration_seconds\"\r\n\r\n8",
							   @"name=\"aspect_ratio\"\r\n\r\n9:16", @"name=\"resolution\"\r\n\r\n1080p",
							   @"name=\"frame_rate\"\r\n\r\n24", @"name=\"negative_prompt\"\r\n\r\nshaky camera",
							   @"name=\"seed\"\r\n\r\n7", @"name=\"person_generation\"\r\n\r\nallow_adult",
							   @"name=\"image\"\r\n\r\n" ]) {
		XCTAssertTrue([form containsString:field], @"%@", field);
	}
	XCTAssertEqual([form componentsSeparatedByString:@"name=\"reference_images[]\""].count, 3, @"each reference is its own field");
	XCTAssertEqualObjects(gemini.jsonRequests[1].URL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/openai/videos/op1");
	XCTAssertEqualObjects(gemini.downloadRequest.URL.host, @"generativelanguage.googleapis.com");
	XCTAssertEqualObjects(gemini.downloadRequest.allHTTPHeaderFields[@"Authorization"], @"Bearer k");
	XCTAssertNotNil([result outputForKey:NFKOutputVideo]);
	CGImageRelease(first);
	CGImageRelease(reference);
}

- (void)testGeminiSoraCompatibleExtendsAClipByItsIdentifierAndRefusesAnEdit
{
	NFKStubVideoBackend *gemini = [self stubForProvider:NFKRemoteProvider.googleGemini style:NFKRemoteVideoAPIStyleGeminiSoraCompatible
												  model:@"veo-3.1-generate-preview"];
	gemini.stagedJSON = [@[ @{ @"id": @"op2", @"status": @"processing" },
							@{ @"id": @"op2", @"status": @"completed", @"url": @"https://generativelanguage.googleapis.com/clip" } ] mutableCopy];
	NSDictionary *extend = @{ NFKParameterVideoOperation: NFKVideoOperationExtend, NFKParameterSourceVideoIdentifier: @"op1" };
	NFKInferenceRequest *extension = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"keep going" } parameters:extend];
	XCTAssertNotNil([gemini runInferenceForRequest:extension error:NULL]);
	XCTAssertTrue([[self multipartTextOf:gemini.jsonRequests.firstObject] containsString:@"name=\"extend_video_id\"\r\n\r\nop1"]);

	NSError *error = nil;
	NSDictionary *edit = @{ NFKParameterVideoOperation: NFKVideoOperationEdit, NFKParameterSourceVideoIdentifier: @"op1" };
	NFKInferenceRequest *editing = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" } parameters:edit];
	XCTAssertNil([gemini runInferenceForRequest:editing error:&error]);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
}

// Gemini's native Veo API: instances and parameters, its own key header, and an operation to poll.
- (void)testGeminiVeoSubmitsInstancesAndParametersAndReturnsEverySample
{
	NFKStubVideoBackend *veo = [self stubForProvider:NFKRemoteProvider.googleGemini style:NFKRemoteVideoAPIStyleGeminiVeo
											   model:@"veo-3.1-generate-preview"];
	NSString *operation = @"models/veo-3.1-generate-preview/operations/abc";
	NSDictionary *sample = @{ @"video": @{ @"uri": @"https://generativelanguage.googleapis.com/v1beta/files/one:download?alt=media" } };
	NSDictionary *second = @{ @"video": @{ @"uri": @"https://generativelanguage.googleapis.com/v1beta/files/two:download?alt=media" } };
	veo.stagedJSON = [@[ @{ @"name": operation },
						 @{ @"name": operation, @"done": @NO },
						 @{ @"name": operation, @"done": @YES,
							@"response": @{ @"generateVideoResponse": @{ @"generatedSamples": @[ sample, second ] } } } ] mutableCopy];
	CGImageRef first = [self newSquare];
	CGImageRef last = [self newSquare];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a bridge", NFKInputImage: (__bridge id)first,
																			  NFKInputLastFrame: (__bridge id)last }
																parameters:@{ NFKParameterWidth: @1920, NFKParameterHeight: @1080,
																			  NFKParameterDurationSeconds: @8, NFKParameterSampleCount: @2,
																			  @"personGeneration": @"allow_adult" }
															outputModality:NFKModalityVideo];
	NSError *error = nil;
	NFKInferenceResult *result = [veo runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);
	NSURLRequest *submit = veo.jsonRequests.firstObject;
	XCTAssertEqualObjects(submit.URL.absoluteString,
						  @"https://generativelanguage.googleapis.com/v1beta/models/veo-3.1-generate-preview:predictLongRunning");
	XCTAssertEqualObjects(submit.allHTTPHeaderFields[@"x-goog-api-key"], @"k");
	XCTAssertNil(submit.allHTTPHeaderFields[@"Authorization"]);
	NSDictionary *body = [self bodyOf:submit];
	NSDictionary *instance = body[@"instances"][0];
	XCTAssertEqualObjects(instance[@"prompt"], @"a bridge");
	XCTAssertEqualObjects(instance[@"image"][@"inlineData"][@"mimeType"], @"image/png");
	XCTAssertNotNil(instance[@"lastFrame"][@"inlineData"][@"data"]);
	NSDictionary *parameters = body[@"parameters"];
	XCTAssertEqualObjects(parameters[@"aspectRatio"], @"16:9", @"derived from the size");
	XCTAssertEqualObjects(parameters[@"resolution"], @"1080p");
	XCTAssertEqualObjects(parameters[@"durationSeconds"], @8);
	XCTAssertEqualObjects(parameters[@"numberOfVideos"], @2);
	XCTAssertEqualObjects(parameters[@"personGeneration"], @"allow_adult");

	XCTAssertEqualObjects(veo.jsonRequests[1].URL.absoluteString,
						  @"https://generativelanguage.googleapis.com/v1beta/models/veo-3.1-generate-preview/operations/abc");
	XCTAssertEqualObjects(veo.jsonRequests[1].allHTTPHeaderFields[@"x-goog-api-key"], @"k");
	XCTAssertEqual(veo.downloadRequests.count, 2);
	XCTAssertEqualObjects(veo.downloadRequests[0].allHTTPHeaderFields[@"x-goog-api-key"], @"k");
	XCTAssertEqual([[result outputForKey:NFKOutputVideos] count], 2);
	CGImageRelease(first);
	CGImageRelease(last);
}

- (void)testGeminiVeoReportsTheOperationsError
{
	NFKStubVideoBackend *veo = [self stubForProvider:NFKRemoteProvider.googleGemini style:NFKRemoteVideoAPIStyleGeminiVeo model:@"veo-3.1-lite-generate-preview"];
	veo.stagedJSON = [@[ @{ @"name": @"models/x/operations/y" },
						 @{ @"name": @"models/x/operations/y", @"done": @YES, @"error": @{ @"code": @3, @"message": @"unsafe prompt" } } ] mutableCopy];
	NSError *error = nil;
	NFKInferenceRequest *prompt = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }];
	XCTAssertNil([veo runInferenceForRequest:prompt error:&error]);
	XCTAssertTrue([error.localizedDescription containsString:@"unsafe prompt"], @"%@", error);

	veo.modelName = nil;
	XCTAssertNil([veo runInferenceForRequest:prompt error:&error]);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported, @"the model is in the path");
}

// xAI: its own paths per operation, a request_id, done / pending, and a clip on a storage host.
- (void)testXAIGeneratesWithARatioAndTierDerivedFromTheSize
{
	NFKStubVideoBackend *xai = [self stubForProvider:NFKRemoteProvider.xAI style:NFKRemoteVideoAPIStyleXAI model:@"grok-imagine-video-1.5"];
	xai.stagedJSON = [@[ @{ @"request_id": @"req1" },
						 @{ @"status": @"pending", @"progress": @50 },
						 @{ @"status": @"done", @"video": @{ @"url": @"https://vidgen.x.ai/bucket/xai-video-req1.mp4", @"duration": @6 } } ] mutableCopy];
	CGImageRef first = [self newSquare];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a lake", NFKInputImage: (__bridge id)first }
																parameters:@{ NFKParameterWidth: @1280, NFKParameterHeight: @720,
																			  NFKParameterDurationSeconds: @6, NFKParameterSeed: @1 }
															outputModality:NFKModalityVideo];
	NSError *error = nil;
	XCTAssertNotNil([xai runInferenceForRequest:request error:&error], @"%@", error);
	NSDictionary *body = [self bodyOf:xai.jsonRequests.firstObject];
	XCTAssertEqualObjects(body[@"duration"], @6);
	XCTAssertEqualObjects(body[@"aspect_ratio"], @"16:9");
	XCTAssertEqualObjects(body[@"resolution"], @"720p");
	XCTAssertTrue([body[@"image"][@"url"] hasPrefix:@"data:image/png;base64,"]);
	XCTAssertNil(body[@"seed"], @"xAI takes no seed");
	XCTAssertEqualObjects(xai.jsonRequests[1].URL.absoluteString, @"https://api.x.ai/v1/videos/req1");
	XCTAssertEqualObjects(xai.downloadRequest.URL.host, @"vidgen.x.ai");
	XCTAssertNil(xai.downloadRequest.allHTTPHeaderFields[@"Authorization"], @"the key stays with the service's own host");
	CGImageRelease(first);
}

- (void)testXAIEditsAndExtendsOnTheirOwnPaths
{
	NFKStubVideoBackend *xai = [self stubForProvider:NFKRemoteProvider.xAI style:NFKRemoteVideoAPIStyleXAI model:@"grok-imagine-video"];
	NFKVideoAsset *hosted = [NFKVideoAsset videoAssetWithFileURL:[NSURL URLWithString:@"https://example.com/clip.mp4"]];
	xai.stagedJSON = [@[ @{ @"request_id": @"r" }, @{ @"status": @"done", @"video": @{ @"url": @"https://vidgen.x.ai/a.mp4" } } ] mutableCopy];
	[xai runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"longer", NFKInputVideo: hosted }
															parameters:@{ NFKParameterVideoOperation: NFKVideoOperationExtend,
																		  NFKParameterDurationSeconds: @4 }] error:NULL];
	XCTAssertEqualObjects(xai.jsonRequests.firstObject.URL.absoluteString, @"https://api.x.ai/v1/videos/extensions");
	NSDictionary *body = [self bodyOf:xai.jsonRequests.firstObject];
	XCTAssertEqualObjects(body[@"video"], (@{ @"url": @"https://example.com/clip.mp4" }));
	XCTAssertEqualObjects(body[@"duration"], @4);

	xai.jsonRequests = nil;
	xai.stagedJSON = [@[ @{ @"request_id": @"r" }, @{ @"status": @"done", @"video": @{ @"url": @"https://vidgen.x.ai/a.mp4" } } ] mutableCopy];
	[xai runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"make it night", NFKInputVideo: hosted }
															parameters:@{ NFKParameterVideoOperation: NFKVideoOperationEdit }] error:NULL];
	XCTAssertEqualObjects(xai.jsonRequests.firstObject.URL.absoluteString, @"https://api.x.ai/v1/videos/edits");
}

// Together: width and height, keyframes under media, and a WebM the bytes identify.
- (void)testTogetherSendsKeyframesUnderMediaAndKeepsAWebMAsWebM
{
	NFKStubVideoBackend *together = [self stubForProvider:NFKRemoteProvider.together style:NFKRemoteVideoAPIStyleTogether model:@"minimax/hailuo-02"];
	const uint8_t webm[] = { 0x1A, 0x45, 0xDF, 0xA3, 0x01, 0x02 };
	together.stagedClip = [NSData dataWithBytes:webm length:sizeof(webm)];
	together.stagedJSON = [@[ @{ @"id": @"v1", @"status": @"queued" },
							  @{ @"id": @"v1", @"status": @"completed", @"outputs": @{ @"video_url": @"https://api.together.ai/shrt/abc", @"cost": @0.28 } } ] mutableCopy];
	CGImageRef first = [self newSquare];
	CGImageRef last = [self newSquare];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a cat", NFKInputImage: (__bridge id)first,
																			  NFKInputLastFrame: (__bridge id)last }
																parameters:@{ NFKParameterWidth: @1366, NFKParameterHeight: @768,
																			  NFKParameterDurationSeconds: @6, NFKParameterSteps: @30,
																			  NFKParameterSeed: @42, NFKParameterGuidanceScale: @7.5,
																			  NFKParameterFramesPerSecond: @24, NFKParameterOutputFormat: @"webm",
																			  @"output_quality": @20 }
															outputModality:NFKModalityVideo];
	NSError *error = nil;
	NFKInferenceResult *result = [together runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);
	NSDictionary *body = [self bodyOf:together.jsonRequests.firstObject];
	XCTAssertEqualObjects(body[@"width"], @1366);
	XCTAssertEqualObjects(body[@"height"], @768);
	XCTAssertEqualObjects(body[@"seconds"], @"6");
	XCTAssertEqualObjects(body[@"steps"], @30);
	XCTAssertEqualObjects(body[@"guidance_scale"], @7.5);
	XCTAssertEqualObjects(body[@"fps"], @24);
	XCTAssertEqualObjects(body[@"output_format"], @"WEBM");
	XCTAssertEqualObjects(body[@"output_quality"], @20);
	NSArray *frames = body[@"media"][@"frame_images"];
	XCTAssertEqualObjects(frames[0][@"frame"], @"first");
	XCTAssertEqualObjects(frames[1][@"frame"], @"last");
	XCTAssertEqualObjects(together.jsonRequests[1].URL.absoluteString, @"https://api.together.xyz/v2/videos/v1");
	XCTAssertEqualObjects([[result outputForKey:NFKOutputVideo] fileURL].pathExtension, @"webm");
	CGImageRelease(first);
	CGImageRelease(last);
}

- (void)testTogetherRefusesALocalSourceClip
{
	NFKStubVideoBackend *together = [self stubForProvider:NFKRemoteProvider.together style:NFKRemoteVideoAPIStyleTogether model:@"Wan-AI/wan2.7-videoedit"];
	NFKVideoAsset *local = [NFKVideoAsset videoAssetWithFileURL:[NSURL fileURLWithPath:@"/tmp/clip.mp4"]];
	NSError *error = nil;
	NFKInferenceRequest *edit = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x", NFKInputVideo: local }
															parameters:@{ NFKParameterVideoOperation: NFKVideoOperationEdit }];
	XCTAssertNil([together runInferenceForRequest:edit error:&error]);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
	XCTAssertEqual(together.jsonRequests.count, 0, @"refused before the submit");
}

// OpenRouter: frame_images, a previous job to extend, and relative content URLs resolved on its host.
- (void)testOpenRouterSendsFramesAndResolvesItsRelativeContentURLs
{
	NFKStubVideoBackend *router = [self stubForProvider:NFKRemoteProvider.openRouter style:NFKRemoteVideoAPIStyleOpenRouter model:@"google/veo-3.1"];
	router.stagedJSON = [@[ @{ @"id": @"gen-vid-1", @"polling_url": @"/api/v1/videos/gen-vid-1", @"status": @"pending" },
							@{ @"id": @"gen-vid-1", @"status": @"completed",
							   @"unsigned_urls": @[ @"/api/v1/videos/gen-vid-1/content?index=0" ] } ] mutableCopy];
	CGImageRef first = [self newSquare];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"mountains", NFKInputImage: (__bridge id)first }
																parameters:@{ NFKParameterWidth: @1280, NFKParameterHeight: @720,
																			  NFKParameterDurationSeconds: @8, NFKParameterGenerateAudio: @YES,
																			  NFKParameterVideoOperation: NFKVideoOperationExtend,
																			  NFKParameterSourceVideoIdentifier: @"gen-vid-0" }
															outputModality:NFKModalityVideo];
	NSError *error = nil;
	XCTAssertNotNil([router runInferenceForRequest:request error:&error], @"%@", error);
	NSDictionary *body = [self bodyOf:router.jsonRequests.firstObject];
	XCTAssertEqualObjects(body[@"size"], @"1280x720");
	XCTAssertEqualObjects(body[@"duration"], @8);
	XCTAssertEqualObjects(body[@"generate_audio"], @YES);
	XCTAssertEqualObjects(body[@"previous_job_id"], @"gen-vid-0");
	XCTAssertEqualObjects(body[@"frame_images"][0][@"frame_type"], @"first_frame");
	XCTAssertEqualObjects(router.jsonRequests[1].URL.absoluteString, @"https://openrouter.ai/api/v1/videos/gen-vid-1");
	XCTAssertEqualObjects(router.downloadRequest.URL.absoluteString, @"https://openrouter.ai/api/v1/videos/gen-vid-1/content?index=0");
	XCTAssertEqualObjects(router.downloadRequest.allHTTPHeaderFields[@"Authorization"], @"Bearer k");
	CGImageRelease(first);
}

- (void)testOpenAIEditsAndExtendsByTheJobsIdentifier
{
	self.backend.stagedJSON = [@[ @{ @"id": @"video_3", @"status": @"completed" } ] mutableCopy];
	[self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"longer" }
																	 parameters:@{ NFKParameterVideoOperation: NFKVideoOperationExtend,
																				   NFKParameterSourceVideoIdentifier: @"video_1",
																				   NFKParameterDurationSeconds: @8 }] error:NULL];
	XCTAssertEqualObjects(self.backend.jsonRequests.firstObject.URL.absoluteString, @"https://api.openai.com/v1/videos/extensions");
	NSDictionary *body = [self bodyOf:self.backend.jsonRequests.firstObject];
	XCTAssertEqualObjects(body[@"video"], (@{ @"id": @"video_1" }));
	XCTAssertEqualObjects(body[@"seconds"], @"8");

	NSError *error = nil;
	NFKInferenceRequest *edit = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }
															parameters:@{ NFKParameterVideoOperation: NFKVideoOperationEdit }];
	XCTAssertNil([self.backend runInferenceForRequest:edit error:&error]);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported, @"an edit needs the earlier job's identifier");
}

- (void)testTheFactoryDerivesTheVideosURLAndDeclinesAnthropic
{
	NFKRemoteVideoBackend *openAI = [NFKRemoteVideoBackend backendForProvider:NFKRemoteProvider.openAI apiKey:@"k" modelName:@"sora-2"];
	XCTAssertEqualObjects(openAI.submitURL.absoluteString, @"https://api.openai.com/v1/videos");
	XCTAssertEqualObjects(openAI.backendIdentifier, @"remote-video");
	XCTAssertEqualWithAccuracy(openAI.pollInterval, 5, 1e-9, @"a clip takes minutes; polling every two seconds is noise");
	XCTAssertNil([NFKRemoteVideoBackend backendForProvider:NFKRemoteProvider.anthropic apiKey:@"k" modelName:@"m"]);
}

@end
