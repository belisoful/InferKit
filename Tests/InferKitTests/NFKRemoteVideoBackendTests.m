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

- (void)testTheFactoryDerivesTheVideosURLAndDeclinesAnthropic
{
	NFKRemoteVideoBackend *openAI = [NFKRemoteVideoBackend backendForProvider:NFKRemoteProvider.openAI apiKey:@"k" modelName:@"sora-2"];
	XCTAssertEqualObjects(openAI.submitURL.absoluteString, @"https://api.openai.com/v1/videos");
	XCTAssertEqualObjects(openAI.backendIdentifier, @"remote-video");
	XCTAssertEqualWithAccuracy(openAI.pollInterval, 5, 1e-9, @"a clip takes minutes; polling every two seconds is noise");
	XCTAssertNil([NFKRemoteVideoBackend backendForProvider:NFKRemoteProvider.anthropic apiKey:@"k" modelName:@"m"]);
}

@end
