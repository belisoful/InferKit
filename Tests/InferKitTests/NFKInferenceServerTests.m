//
//  NFKInferenceServerTests.m
//  InferKitTests
//
//  Every test starts a real server on a free loopback port and drives it with the core's own
//  clients, unchanged, so what is asserted is the round trip a machine on the network makes.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreML/CoreML.h>
#import <ifaddrs.h>
#import <net/if.h>
#import <arpa/inet.h>
#import <netinet/in.h>
#import <sys/socket.h>
#import <unistd.h>

/*! A synchronous backend whose reply a test writes, recording what reached it. */
@interface NFKServedTestBackend : NSObject <NFKInferenceBackend>
@property (atomic, strong, nullable) NFKInferenceRequest *lastRequest;
@property (nonatomic, copy) NFKInferenceResult * _Nullable (^reply)(NFKInferenceRequest *request, NSError **error);
@property (atomic, assign) NSInteger running;
@property (atomic, assign) NSInteger mostRunning;
@property (nonatomic, assign) NSTimeInterval delay;
@end

@implementation NFKServedTestBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"served-test";
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithArray:@[ NFKInputPrompt, NFKInputMessages ]];
}

- (NSSet<NSString *> *)supportedParameterKeys
{
	return [NSSet setWithObject:NFKParameterTemperature];
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	@synchronized (self) {
		self.running += 1;
		self.mostRunning = MAX(self.mostRunning, self.running);
	}
	self.lastRequest = request;
	if (self.delay > 0) {
		[NSThread sleepForTimeInterval:self.delay];
	}
	NFKInferenceResult *result = self.reply != nil ? self.reply(request, error)
		: [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"ok" }];
	@synchronized (self) {
		self.running -= 1;
	}
	return result;
}

@end

/*! A backend that streams: it reports each partial text, then finishes with the last. */
@interface NFKServedStreamingBackend : NFKServedTestBackend
@property (nonatomic, copy) NSArray<NSString *> *partials;
@end

@implementation NFKServedStreamingBackend

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	self.lastRequest = request;
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	NSArray<NSString *> *partials = self.partials;
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		for (NSString *text in partials) {
			[job reportProgress:0.5 partialResult:[NFKInferenceResult resultWithOutputs:@{ NFKOutputText: text }]];
			[NSThread sleepForTimeInterval:0.02];
		}
		[job finishWithResult:[NFKInferenceResult resultWithOutputs:@{ NFKOutputText: partials.lastObject ?: @"" }]];
	});
	return job;
}

@end

/*! A backend whose job streams one piece of text and then fails as rate limited. */
@interface NFKServedFailingStreamBackend : NFKServedTestBackend
@end

@implementation NFKServedFailingStreamBackend

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		[job reportProgress:0.5 partialResult:[NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"partial" }]];
		[NSThread sleepForTimeInterval:0.05];
		[job finishWithError:[NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceRateLimited
											 userInfo:@{ NSLocalizedDescriptionKey: @"out of quota" }]];
	});
	return job;
}

@end

/*! A backend whose job runs until it is cancelled, and says so. */
@interface NFKServedHangingBackend : NFKServedTestBackend
@property (nonatomic, strong) XCTestExpectation *cancelled;
@property (atomic, assign) BOOL started;
@end

@implementation NFKServedHangingBackend

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	self.started = YES;
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	XCTestExpectation *cancelled = self.cancelled;
	job.cancellationHandler = ^{
		[cancelled fulfill];
	};
	return job;
}

@end

/*! A hanging job that reports a quarter of its progress shortly after it starts. */
@interface NFKServedProgressingBackend : NFKServedHangingBackend
@end

@implementation NFKServedProgressingBackend

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [super submitInferenceJobForRequest:request];
	// After the server has installed its progress handler, which happens once this returns.
	dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.05 * NSEC_PER_SEC)), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		[job reportProgress:0.25];
	});
	return job;
}

@end

/*! A backend describing its model, with two values a JSON reply cannot carry. */
@interface NFKServedDescribedBackend : NFKServedTestBackend
@end

@implementation NFKServedDescribedBackend

- (NSDictionary<NSString *, id> *)modelInfo
{
	return @{ NFKModelInfoParameterCount: @7000000, NFKModelInfoPrecision: @"bfloat16",
			  @"loaded_at": [NSDate date], @"unmeasured": @(NAN) };
}

@end

/*! A runtime that reports a fixed state. */
@interface NFKServedTestRuntime : NSObject <NFKServingRuntimeStatus>
@end

@implementation NFKServedTestRuntime

+ (NSString *)runtimeStatusName
{
	return @"test-runtime";
}

+ (NSDictionary<NSString *, id> *)runtimeStatus
{
	return @{ @"active_memory_bytes": @4096 };
}

@end

@interface NFKInferenceServerTests : XCTestCase
@property (nonatomic, assign) NSUInteger savedRetryAttempts;
@end

@implementation NFKInferenceServerTests

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
{
	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.port = 0;
	server.loopbackOnly = YES;
	for (NSString *name in backends) {
		[server addBackend:backends[name] forModelName:name];
	}
	NSError *error = nil;
	XCTAssertTrue([server startWithError:&error], @"%@", error);
	[self addTeardownBlock:^{
		[server stop];
	}];
	return server;
}

- (NFKRemoteProvider *)providerFor:(NFKInferenceServer *)server
{
	return [NFKRemoteProvider.inferKit providerWithBaseURL:server.localBaseURL];
}

- (nullable NSData *)send:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse
{
	return [NFKRemoteTransport sendRequest:request session:NSURLSession.sharedSession response:outResponse error:NULL];
}

- (NSMutableURLRequest *)postTo:(NSURL *)url JSON:(id)body
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.HTTPMethod = @"POST";
	request.HTTPBody = [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL];
	[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	return request;
}

- (CVPixelBufferRef)pixelBufferWidth:(size_t)width height:(size_t)height format:(OSType)format CF_RETURNS_RETAINED
{
	CVPixelBufferRef buffer = NULL;
	NSDictionary *attributes = @{ (id)kCVPixelBufferIOSurfacePropertiesKey: @{} };
	CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, (__bridge CFDictionaryRef)attributes, &buffer);
	CVPixelBufferLockBaseAddress(buffer, 0);
	uint8_t *base = CVPixelBufferGetBaseAddress(buffer);
	size_t bytesPerRow = CVPixelBufferGetBytesPerRow(buffer);
	for (size_t row = 0; row < height; row++) {
		for (size_t column = 0; column < bytesPerRow; column++) {
			base[row * bytesPerRow + column] = (uint8_t)((row * 31 + column * 7) & 0xFF);
		}
	}
	CVPixelBufferUnlockBaseAddress(buffer, 0);
	return buffer;
}

- (BOOL)pixelBuffer:(CVPixelBufferRef)left matches:(CVPixelBufferRef)right
{
	if (CVPixelBufferGetWidth(left) != CVPixelBufferGetWidth(right) || CVPixelBufferGetHeight(left) != CVPixelBufferGetHeight(right)
		|| CVPixelBufferGetPixelFormatType(left) != CVPixelBufferGetPixelFormatType(right)) {
		return NO;
	}
	CVPixelBufferLockBaseAddress(left, kCVPixelBufferLock_ReadOnly);
	CVPixelBufferLockBaseAddress(right, kCVPixelBufferLock_ReadOnly);
	size_t rowBytes = MIN(CVPixelBufferGetBytesPerRow(left), CVPixelBufferGetBytesPerRow(right));
	size_t used = CVPixelBufferGetWidth(left) * (CVPixelBufferGetBytesPerRow(left) / MAX(CVPixelBufferGetWidth(left), (size_t)1));
	BOOL same = YES;
	for (size_t row = 0; row < CVPixelBufferGetHeight(left) && same; row++) {
		same = memcmp((uint8_t *)CVPixelBufferGetBaseAddress(left) + row * CVPixelBufferGetBytesPerRow(left),
					  (uint8_t *)CVPixelBufferGetBaseAddress(right) + row * CVPixelBufferGetBytesPerRow(right), MIN(rowBytes, used)) == 0;
	}
	CVPixelBufferUnlockBaseAddress(left, kCVPixelBufferLock_ReadOnly);
	CVPixelBufferUnlockBaseAddress(right, kCVPixelBufferLock_ReadOnly);
	return same;
}

- (NSURL *)writtenToneWithExtension:(NSString *)extension
{
	NSURL *url = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:
				  [NSString stringWithFormat:@"served-tone-%@.%@", NSUUID.UUID.UUIDString, extension]];
	AVAudioFormat *format = [[AVAudioFormat alloc] initStandardFormatWithSampleRate:16000 channels:1];
	AVAudioPCMBuffer *buffer = [[AVAudioPCMBuffer alloc] initWithPCMFormat:format frameCapacity:8000];
	buffer.frameLength = 8000;
	for (AVAudioFrameCount frame = 0; frame < 8000; frame++) {
		buffer.floatChannelData[0][frame] = 0.25f * sinf(2.0f * (float)M_PI * 440.0f * (float)frame / 16000.0f);
	}
	NSDictionary *settings = @{ AVFormatIDKey: @(kAudioFormatLinearPCM), AVSampleRateKey: @16000, AVNumberOfChannelsKey: @1,
								AVLinearPCMBitDepthKey: @16, AVLinearPCMIsFloatKey: @NO, AVLinearPCMIsBigEndianKey: @NO };
	@autoreleasepool {
		AVAudioFile *file = [[AVAudioFile alloc] initForWriting:url settings:settings error:NULL];
		[file writeFromBuffer:buffer error:NULL];
	}
	[self addTeardownBlock:^{
		[NSFileManager.defaultManager removeItemAtURL:url error:NULL];
	}];
	return url;
}

#pragma mark Configuration

- (void)testAServerBeyondLoopbackRefusesToStartWithoutAKey
{
	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.port = 0;
	server.advertisesService = NO;
	NSError *error = nil;
	XCTAssertFalse([server startWithError:&error]);
	XCTAssertEqualObjects(error.domain, NFKInferenceServerErrorDomain);
	XCTAssertEqual(error.code, NFKInferenceServerErrorAPIKeyRequired);
	XCTAssertFalse(server.isRunning);

	server.requiresAPIKey = NO;
	XCTAssertTrue([server startWithError:&error], @"a host may choose to serve without a key: %@", error);
	[server stop];

	NFKInferenceServer *local = [[NFKInferenceServer alloc] init];
	local.port = 0;
	local.loopbackOnly = YES;
	XCTAssertTrue([local startWithError:&error], @"loopback needs no key: %@", error);
	XCTAssertEqualObjects(local.localBaseURL.host, @"localhost");
	XCTAssertEqualObjects(local.localBaseURL.path, @"/v1");
	XCTAssertGreaterThan(local.listeningPort, 0);
	[local stop];
	XCTAssertNil(local.localBaseURL);
}

- (void)testATakenPortFailsToListen
{
	NFKInferenceServer *first = [self startedServerHosting:@{}];
	NFKInferenceServer *second = [[NFKInferenceServer alloc] init];
	second.loopbackOnly = YES;
	second.port = first.listeningPort;
	NSError *error = nil;
	XCTAssertFalse([second startWithError:&error]);
	XCTAssertEqualObjects(error.domain, NFKInferenceServerErrorDomain);
	XCTAssertEqual(error.code, NFKInferenceServerErrorListenFailed);
}

#pragma mark Chat

- (void)testTheChatClientRunsAHostedBackendWithItsParametersRenamedBack
{
	NFKServedTestBackend *backend = [[NFKServedTestBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"hello from the server",
														NFKOutputUsage: @{ NFKUsageInputTokens: @7, NFKUsageOutputTokens: @4 } }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"served-chat": backend }];
	id<NFKInferenceBackend> client = [NFKRemoteBackend backendWithEndpointURL:[[self providerFor:server] URLForPath:@"chat/completions"]];
	[(NFKRemoteBackend *)client setModelName:@"served-chat"];

	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }
															   parameters:@{ NFKParameterMaxTokens: @12, NFKParameterTemperature: @0.5,
																			 NFKParameterStopSequences: @[ @"END" ],
																			 NFKParameterReasoningEffort: NFKReasoningEffortDeep,
																			 NFKParameterConversationKey: @"chat-7",
																			 @"custom_field": @"kept" }
														   outputModality:NFKModalityText];
	NSError *error = nil;
	NFKInferenceResult *result = [client runInferenceForRequest:request error:&error];
	XCTAssertEqualObjects(result.text, @"hello from the server", @"%@", error);
	XCTAssertEqualObjects([result outputForKey:NFKOutputUsage][NFKUsageOutputTokens], @4);

	NFKInferenceRequest *received = backend.lastRequest;
	XCTAssertEqualObjects(received.prompt, @"hi", @"a lone user turn also arrives as the prompt");
	XCTAssertEqualObjects(received.messages.firstObject[@"content"], @"hi");
	XCTAssertEqualObjects([received parameterForKey:NFKParameterMaxTokens], @12);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterTemperature], @0.5);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterStopSequences], (@[ @"END" ]));
	XCTAssertEqualObjects([received parameterForKey:NFKParameterReasoningEffort], NFKReasoningEffortDeep);
	XCTAssertEqualObjects([received parameterForKey:NFKParameterConversationKey], @"chat-7", @"read back from prompt_cache_key");
	XCTAssertEqualObjects([received parameterForKey:@"custom_field"], @"kept");
	XCTAssertEqual(received.outputModality, NFKModalityText);
}

- (void)testTheChatClientStreamsTheHostedJobsPartialText
{
	NFKServedStreamingBackend *backend = [[NFKServedStreamingBackend alloc] init];
	backend.partials = @[ @"Hel", @"Hello", @"Hello, world" ];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"streamer": backend }];
	NFKRemoteBackend *client = [NFKRemoteBackend backendWithEndpointURL:[[self providerFor:server] URLForPath:@"chat/completions"]];

	NSMutableArray<NSString *> *seen = [NSMutableArray array];
	XCTestExpectation *done = [self expectationWithDescription:@"stream"];
	NFKInferenceJob *job = [client submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"go" }
																							parameters:nil outputModality:NFKModalityText]];
	job.progressHandler = ^(NFKInferenceJob *progressed) {
		@synchronized (seen) {
			if (progressed.partialResult.text != nil) {
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
	XCTAssertGreaterThan(seen.count, 1, @"the text arrived in more than one piece");
	XCTAssertEqualObjects(seen.lastObject, @"Hello, world");
}

// Once the first delta is out the status is spent, so a failure after it rides as an error event,
// and the client reports that error rather than the partial text as an answer.
- (void)testAStreamThatFailsPartwayFailsTheClientsJob
{
	NFKServedFailingStreamBackend *failing = [[NFKServedFailingStreamBackend alloc] init];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"flaky": failing }];
	NFKRemoteBackend *client = [NFKRemoteBackend backendWithEndpointURL:[[self providerFor:server] URLForPath:@"chat/completions"]];
	XCTestExpectation *done = [self expectationWithDescription:@"stream"];
	NFKInferenceJob *job = [client submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"go" }
																							parameters:nil outputModality:NFKModalityText]];
	job.completionHandler = ^(NFKInferenceJob *finished) {
		[done fulfill];
	};
	[self waitForExpectations:@[ done ] timeout:10];
	XCTAssertEqual(job.status, NFKInferenceJobStatusFailed);
	XCTAssertEqual(job.error.code, kNFKError_InferenceRateLimited, @"%@", job.error);
	XCTAssertTrue([job.error.localizedDescription containsString:@"out of quota"]);
}

- (void)testToolsGoOutAsTheContractShapeAndCallsComeBack
{
	NFKServedTestBackend *backend = [[NFKServedTestBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputToolCalls: @[ @{ @"id": @"call_1", @"name": @"get_weather",
																				  @"arguments": @{ @"city": @"Paris" } } ] }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"tools": backend }];
	NFKRemoteBackend *client = [NFKRemoteBackend backendWithEndpointURL:[[self providerFor:server] URLForPath:@"chat/completions"]];
	NSDictionary *tool = @{ @"name": @"get_weather", @"description": @"weather",
							@"parameters": @{ @"type": @"object", @"properties": @{ @"city": @{ @"type": @"string" } } } };
	NSDictionary *schema = @{ @"type": @"object" };
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"weather?" }
															   parameters:@{ NFKParameterTools: @[ tool ], NFKParameterJSONSchema: schema }
														   outputModality:NFKModalityText];
	NSError *error = nil;
	NFKInferenceResult *result = [client runInferenceForRequest:request error:&error];
	XCTAssertEqualObjects(result.toolCalls.firstObject[@"name"], @"get_weather", @"%@", error);
	XCTAssertEqualObjects(result.toolCalls.firstObject[@"arguments"], (@{ @"city": @"Paris" }));
	XCTAssertEqualObjects(result.toolCalls.firstObject[@"id"], @"call_1");
	XCTAssertEqualObjects([backend.lastRequest parameterForKey:NFKParameterTools], (@[ tool ]), @"the wrapper is removed");
	XCTAssertEqualObjects([backend.lastRequest parameterForKey:NFKParameterJSONSchema], schema);
}

- (void)testAnImageInTheChatArrivesAsAPixelBuffer
{
	NFKServedTestBackend *backend = [[NFKServedTestBackend alloc] init];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"vision": backend }];
	NFKRemoteBackend *client = [NFKRemoteBackend backendWithEndpointURL:[[self providerFor:server] URLForPath:@"chat/completions"]];
	CVPixelBufferRef image = [self pixelBufferWidth:6 height:3 format:kCVPixelFormatType_32BGRA];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"what is it?", NFKInputImage: (__bridge id)image }
															   parameters:nil outputModality:NFKModalityText];
	NSError *error = nil;
	XCTAssertNotNil([client runInferenceForRequest:request error:&error], @"%@", error);
	CVPixelBufferRef received = (__bridge CVPixelBufferRef)[backend.lastRequest inputForKey:NFKInputImage];
	XCTAssertTrue(received != NULL && CFGetTypeID(received) == CVPixelBufferGetTypeID());
	XCTAssertEqual(CVPixelBufferGetWidth(received), 6u);
	XCTAssertEqual(CVPixelBufferGetHeight(received), 3u);
	XCTAssertEqualObjects(backend.lastRequest.messages.lastObject[@"content"], @"what is it?", @"the text part is the turn's content");
	CVPixelBufferRelease(image);
}

- (void)testARunErrorReachesTheChatClientWithItsCode
{
	NFKServedTestBackend *backend = [[NFKServedTestBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceMissingInput
								 userInfo:@{ NSLocalizedDescriptionKey: @"no prompt at all" }];
		return nil;
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"failing": backend }];
	NFKRemoteBackend *client = [NFKRemoteBackend backendWithEndpointURL:[[self providerFor:server] URLForPath:@"chat/completions"]];
	NSError *error = nil;
	XCTAssertNil([client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput, @"the body's code outranks the 400's reading: %@", error);
	XCTAssertEqualObjects(error.userInfo[NFKRemoteErrorStatusCodeKey], @400);
	XCTAssertTrue([error.localizedDescription containsString:@"no prompt at all"]);
}

#pragma mark Other OpenAI routes

- (void)testEmbeddingsComeBackInInputOrder
{
	NFKServedTestBackend *backend = [[NFKServedTestBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputEmbedding: @[ @(request.prompt.length), @0.5 ] }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"embedder": backend }];
	NFKRemoteEmbeddingBackend *client = [NFKRemoteEmbeddingBackend backendForProvider:[self providerFor:server] apiKey:nil modelName:@"embedder"];
	XCTAssertNotNil(client, @"an InferKit server serves embeddings");
	NSError *error = nil;
	NSArray<NSArray<NSNumber *> *> *vectors = [client embeddingsForTexts:@[ @"a", @"abc" ] error:&error];
	XCTAssertEqualObjects(vectors, (@[ @[ @1, @0.5 ], @[ @3, @0.5 ] ]), @"%@", error);

	NSHTTPURLResponse *response = nil;
	NSData *data = [self send:[self postTo:[[self providerFor:server] URLForPath:@"embeddings"]
									  JSON:@{ @"input": @"abcd", @"encoding_format": @"base64" }] response:&response];
	NSDictionary *body = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	NSData *floats = [[NSData alloc] initWithBase64EncodedString:body[@"data"][0][@"embedding"] options:0];
	XCTAssertEqual(floats.length, 2 * sizeof(float));
	XCTAssertEqual(((const float *)floats.bytes)[0], 4.0f, @"base64 is little-endian float32, what OpenAI SDKs decode");
}

- (void)testTranscriptionUploadsTheClipAndReadsTheSegments
{
	NFKServedTestBackend *backend = [[NFKServedTestBackend alloc] init];
	__block NSData *uploaded = nil;
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		uploaded = [NSData dataWithContentsOfURL:[(NFKAudioAsset *)[request inputForKey:NFKInputAudio] fileURL]];
		NFKAudioSegment *segment = [NFKAudioSegment segmentWithStartSeconds:0 endSeconds:0.5 label:@"a tone" confidence:1];
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"a tone", NFKOutputSegments: @[ segment ] }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"whisper": backend }];
	NFKRemoteTranscriptionBackend *client = [NFKRemoteTranscriptionBackend backendForProvider:[self providerFor:server] apiKey:nil modelName:@"whisper"];
	client.emitsTimestamps = YES;
	NSURL *tone = [self writtenToneWithExtension:@"wav"];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: [NFKAudioAsset audioAssetWithFileURL:tone] }
															   parameters:@{ NFKParameterSourceLanguage: @"en" } outputModality:NFKModalityText];
	NSError *error = nil;
	NFKInferenceResult *result = [client runInferenceForRequest:request error:&error];
	XCTAssertEqualObjects(result.text, @"a tone", @"%@", error);
	XCTAssertEqual(result.segments.count, 1u);
	XCTAssertEqualWithAccuracy(result.segments.firstObject.endSeconds, 0.5, 1e-9);
	XCTAssertEqualObjects(uploaded, [NSData dataWithContentsOfURL:tone], @"the clip crossed byte for byte");
	XCTAssertEqualObjects([backend.lastRequest parameterForKey:NFKParameterSourceLanguage], @"en");
}

- (void)testSpeechReturnsTheHostedClipAndConvertsItsContainer
{
	NSURL *tone = [self writtenToneWithExtension:@"wav"];
	NFKServedTestBackend *backend = [[NFKServedTestBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputAudio: [NFKAudioAsset audioAssetWithFileURL:tone] }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"voice": backend }];
	NFKRemoteSpeechBackend *client = [NFKRemoteSpeechBackend backendForProvider:[self providerFor:server] apiKey:nil modelName:@"voice" voice:nil];
	XCTAssertFalse(client.requiresVoice, @"the hosted backend chooses its own voice");
	NSError *error = nil;
	NFKInferenceResult *result = [client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"say it" }] error:&error];
	NFKAudioAsset *audio = [result outputForKey:NFKOutputAudio];
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:audio.fileURL], [NSData dataWithContentsOfURL:tone], @"%@", error);
	XCTAssertEqualObjects(backend.lastRequest.prompt, @"say it");

	client.responseFormat = @"flac";
	result = [client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"again" }] error:&error];
	audio = [result outputForKey:NFKOutputAudio];
	AVAudioFile *decoded = [[AVAudioFile alloc] initForReading:audio.fileURL error:&error];
	XCTAssertEqual(decoded.length, 8000, @"the clip was re-encoded as FLAC and still decodes: %@", error);
	XCTAssertEqual(decoded.fileFormat.streamDescription->mFormatID, (AudioFormatID)kAudioFormatFLAC);

	client.responseFormat = @"mp3";
	XCTAssertNil([client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"mp3" }] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported, @"no MP3 encoder ships with the OS: %@", error);
}

- (void)testImageGenerationMapsTheSizeAndReturnsThePicture
{
	NFKServedTestBackend *backend = [[NFKServedTestBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		CVPixelBufferRef image = [self pixelBufferWidth:[[request parameterForKey:NFKParameterWidth] unsignedIntegerValue]
												 height:[[request parameterForKey:NFKParameterHeight] unsignedIntegerValue]
												 format:kCVPixelFormatType_32BGRA];
		NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{ NFKOutputImage: (__bridge id)image }];
		CVPixelBufferRelease(image);
		return result;
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"painter": backend }];
	NFKRemoteImageBackend *client = [NFKRemoteImageBackend backendForProvider:[self providerFor:server] apiKey:nil modelName:@"painter"];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a square" }
															   parameters:@{ NFKParameterWidth: @8, NFKParameterHeight: @4, NFKParameterSeed: @42 }];
	NSError *error = nil;
	NFKInferenceResult *result = [client runInferenceForRequest:request error:&error];
	CVPixelBufferRef image = (__bridge CVPixelBufferRef)[result outputForKey:NFKOutputImage];
	XCTAssertTrue(image != NULL, @"%@", error);
	XCTAssertEqual(CVPixelBufferGetWidth(image), 8u);
	XCTAssertEqual(CVPixelBufferGetHeight(image), 4u);
	XCTAssertEqualObjects([backend.lastRequest parameterForKey:NFKParameterSeed], @42);
	XCTAssertEqual(backend.lastRequest.outputModality, NFKModalityImage);
}

#pragma mark Native route

- (void)testTheNativeRouteCarriesEveryValueTypeExactly
{
	NFKServedTestBackend *backend = [[NFKServedTestBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		NSMutableDictionary *echoed = [request.inputs mutableCopy];
		[echoed addEntriesFromDictionary:request.parameters];
		return [NFKInferenceResult resultWithOutputs:echoed];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"echo": backend }];
	id<NFKInferenceBackend> client = [NFKRemoteProvider backendForProvider:[self providerFor:server] apiKey:nil modelName:@"echo"];
	XCTAssertTrue([client isKindOfClass:NFKRemoteInferKitBackend.class]);

	CVPixelBufferRef depth = [self pixelBufferWidth:5 height:3 format:kCVPixelFormatType_OneComponent32Float];
	CVPixelBufferRef colour = [self pixelBufferWidth:4 height:4 format:kCVPixelFormatType_32BGRA];
	MLMultiArray *array = [[MLMultiArray alloc] initWithShape:@[ @2, @3 ] dataType:MLMultiArrayDataTypeFloat32 error:NULL];
	for (NSInteger index = 0; index < 6; index++) {
		array[index] = @(index * 1.5);
	}
	NFKDetection *detection = [NFKDetection detectionWithLabel:@"cat" classIndex:3 confidence:0.9 boundingBox:CGRectMake(0.1, 0.2, 0.3, 0.4)];
	NFKAudioAsset *clip = [NFKAudioAsset audioAssetWithFileURL:[self writtenToneWithExtension:@"wav"] durationSeconds:0.5 sampleRate:16000 channelCount:1];
	NSDictionary *inputs = @{ @"depth": (__bridge id)depth, @"colour": (__bridge id)colour, @"tensor": array,
							  @"detections": @[ detection ], @"clip": clip, @"bytes": [NSData dataWithBytes:"\x00\x01\xff" length:3],
							  @"tagged": @{ @"$nfk": @"not a tag", @"n": @1 }, @"when": [NSDate dateWithTimeIntervalSince1970:1000] };
	NSArray<NSNumber *> *doubles = @[ @0.1, @(1.0 / 3.0), @(M_PI), @(-2.5e-300) ];
	NSArray<NSNumber *> *singles = @[ @0.1f, @(1.0f / 3.0f) ];
	NSDictionary *parameters = @{ @"nan": @(NAN), @"flag": @YES, @"list": @[ @1, @"two", NSNull.null ],
								  @"doubles": doubles, @"singles": singles, @"third": @(1.0 / 3.0), @"big": @(9007199254740993LL) };
	NSError *error = nil;
	NFKInferenceResult *result = [client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:inputs parameters:parameters
																						outputModality:NFKModalityVideo] error:&error];
	XCTAssertNotNil(result, @"%@", error);
	XCTAssertEqual(backend.lastRequest.outputModality, NFKModalityVideo);
	XCTAssertTrue([self pixelBuffer:depth matches:(__bridge CVPixelBufferRef)[result outputForKey:@"depth"]], @"a float map crosses exactly");
	XCTAssertTrue([self pixelBuffer:colour matches:(__bridge CVPixelBufferRef)[result outputForKey:@"colour"]]);
	MLMultiArray *tensor = [result outputForKey:@"tensor"];
	XCTAssertEqualObjects(tensor.shape, (@[ @2, @3 ]));
	XCTAssertEqualObjects(tensor[5], @7.5);
	XCTAssertEqualObjects([result outputForKey:@"detections"], (@[ detection ]));
	NFKAudioAsset *returned = [result outputForKey:@"clip"];
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:returned.fileURL], [NSData dataWithContentsOfURL:clip.fileURL]);
	XCTAssertEqual(returned.sampleRate, 16000);
	XCTAssertEqualObjects([result outputForKey:@"bytes"], [NSData dataWithBytes:"\x00\x01\xff" length:3]);
	XCTAssertEqualObjects([result outputForKey:@"tagged"], (@{ @"$nfk": @"not a tag", @"n": @1 }), @"a dictionary that looks like a tag stays one");
	XCTAssertEqualObjects([result outputForKey:@"when"], [NSDate dateWithTimeIntervalSince1970:1000]);
	XCTAssertTrue(isnan([[result outputForKey:@"nan"] doubleValue]));
	XCTAssertEqualObjects([result outputForKey:@"flag"], @YES);
	XCTAssertEqualObjects([result outputForKey:@"list"], (@[ @1, @"two", NSNull.null ]));
	for (NSString *name in @[ @"doubles", @"singles" ]) {
		NSArray<NSNumber *> *sent = [name isEqualToString:@"doubles"] ? doubles : singles;
		NSArray<NSNumber *> *received = [result outputForKey:name];
		XCTAssertEqual(received.count, sent.count);
		for (NSUInteger index = 0; index < sent.count; index++) {
			XCTAssertEqual(received[index].doubleValue, sent[index].doubleValue, @"%@[%lu] crosses bit for bit", name, (unsigned long)index);
		}
	}
	XCTAssertEqual([[result outputForKey:@"third"] doubleValue], 1.0 / 3.0);
	XCTAssertEqual([[result outputForKey:@"big"] longLongValue], 9007199254740993LL);
	CVPixelBufferRelease(depth);
	CVPixelBufferRelease(colour);
}

- (void)testTheNativeStreamReportsPartialsAndTheResult
{
	NFKServedStreamingBackend *backend = [[NFKServedStreamingBackend alloc] init];
	backend.partials = @[ @"a", @"ab", @"abc" ];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"streamer": backend }];
	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:server.localBaseURL];
	NSMutableArray<NSString *> *seen = [NSMutableArray array];
	XCTestExpectation *done = [self expectationWithDescription:@"native stream"];
	NFKInferenceJob *job = [client submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"go" }]];
	job.progressHandler = ^(NFKInferenceJob *progressed) {
		@synchronized (seen) {
			if (progressed.partialResult.text != nil && ![seen.lastObject isEqualToString:progressed.partialResult.text]) {
				[seen addObject:progressed.partialResult.text];
			}
		}
	};
	job.completionHandler = ^(NFKInferenceJob *finished) {
		[done fulfill];
	};
	[self waitForExpectations:@[ done ] timeout:10];
	XCTAssertEqualObjects(job.result.text, @"abc", @"%@", job.error);
	XCTAssertEqualObjects(seen, (@[ @"a", @"ab", @"abc" ]));
}

- (void)testTheNativeClientReceivesTheHostedErrorIntact
{
	NFKServedTestBackend *backend = [[NFKServedTestBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceRefused
								 userInfo:@{ NSLocalizedDescriptionKey: @"the guardrail declined", @"reason": @"policy" }];
		return nil;
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"strict": backend }];
	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:server.localBaseURL];
	NSError *error = nil;
	XCTAssertNil([client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }] error:&error]);
	XCTAssertEqualObjects(error.domain, NFKInferenceErrorDomain);
	XCTAssertEqual(error.code, kNFKError_InferenceRefused);
	XCTAssertEqualObjects(error.localizedDescription, @"the guardrail declined");
	XCTAssertEqualObjects(error.userInfo[@"reason"], @"policy");

	client.modelName = @"absent";
	XCTAssertNil([client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }] error:&error]);
	XCTAssertEqualObjects(error.domain, NFKInferenceServerErrorDomain);
	XCTAssertEqual(error.code, NFKInferenceServerErrorModelNotFound);
	XCTAssertEqualObjects(error.userInfo[NFKRemoteErrorStatusCodeKey], @404);
}

// A client names file extensions and shapes. An extension becomes letters and digits, and a shape
// or a size that asks for more memory than the body carried is refused before anything is allocated.
- (void)testWhatAClientNamesCannotReachPastTheBody
{
	NFKServedTestBackend *backend = [[NFKServedTestBackend alloc] init];
	__block NSURL *written = nil;
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		written = [(NFKAudioAsset *)[request inputForKey:NFKInputAudio] fileURL];
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"ok" }];
	};
	NFKInferenceServer *server = [self startedServerHosting:@{ @"guarded": backend }];
	NSDictionary *audioPart = @{ @"type": @"input_audio", @"input_audio": @{ @"data": @"AAAA", @"format": @"../../escaped" } };
	NSDictionary *chat = @{ @"messages": @[ @{ @"role": @"user", @"content": @[ @{ @"type": @"text", @"text": @"hi" }, audioPart ] } ] };
	NSHTTPURLResponse *response = nil;
	[self send:[self postTo:[server.localBaseURL URLByAppendingPathComponent:@"chat/completions"] JSON:chat] response:&response];
	XCTAssertEqual(response.statusCode, 200);
	XCTAssertEqualObjects(written.pathExtension, @"bin");
	XCTAssertEqualObjects(written.URLByDeletingLastPathComponent.lastPathComponent, @"InferKit", @"the file stayed in the temporary directory");

	NSURL *run = [server.localBaseURL URLByAppendingPathComponent:@"inferkit/run"];
	NSDictionary *hugeArray = @{ @"$nfk": @"multiArray", @"dataType": @(MLMultiArrayDataTypeFloat32),
								 @"shape": @[ @100000, @100000, @100000 ], @"base64": @"AAAAAA==" };
	NSDictionary *hugeImage = @{ @"$nfk": @"pixelBuffer", @"format": @(kCVPixelFormatType_32BGRA), @"width": @1000000, @"height": @1000000,
								 @"planes": @[ @{ @"bytesPerRow": @4, @"rows": @1, @"base64": @"AAAAAA==" } ] };
	for (NSDictionary *value in @[ hugeArray, hugeImage ]) {
		NSData *data = [self send:[self postTo:run JSON:@{ @"request": @{ @"inputs": @{ @"value": value } } }] response:&response];
		XCTAssertEqual(response.statusCode, 400, @"%@", data != nil ? [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] : @"");
	}
}

- (void)testPrepareReadsTheHostedBackendsKeys
{
	NFKServedTestBackend *backend = [[NFKServedTestBackend alloc] init];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"only": backend }];
	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:server.localBaseURL];
	XCTAssertFalse([client respondsToSelector:@selector(supportedInputKeys)], @"nothing is declared before prepare");
	NSError *error = nil;
	XCTAssertTrue(NFKInferencePrepare(client, &error), @"%@", error);
	XCTAssertTrue([client respondsToSelector:@selector(supportedInputKeys)]);
	XCTAssertEqualObjects(client.supportedInputKeys, backend.supportedInputKeys);
	XCTAssertEqualObjects(client.supportedParameterKeys, backend.supportedParameterKeys);
}

#pragma mark Models and access

- (void)testTheCatalogListsTheHostedModels
{
	NFKInferenceServer *server = [self startedServerHosting:@{ @"b-model": [[NFKServedTestBackend alloc] init],
															  @"a-model": [[NFKServedTestBackend alloc] init] }];
	NSError *error = nil;
	NSArray<NFKRemoteModel *> *models = [[self providerFor:server] modelsWithAPIKey:nil error:&error];
	XCTAssertEqualObjects([models valueForKey:@"identifier"], (@[ @"a-model", @"b-model" ]), @"%@", error);
	XCTAssertEqualObjects(models.firstObject.ownedBy, @"inferkit");

	// With two models hosted, a request must name one.
	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:server.localBaseURL];
	XCTAssertNil([client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }] error:&error]);
	XCTAssertEqual(error.code, NFKInferenceServerErrorModelNotFound);
}

- (void)testAKeyIsCheckedWhenRequired
{
	NFKServedTestBackend *backend = [[NFKServedTestBackend alloc] init];
	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.port = 0;
	server.loopbackOnly = YES;
	server.requiresAPIKeyOnLoopback = YES;
	server.apiKey = @"sesame";
	[server addBackend:backend forModelName:@"guarded"];
	NSError *error = nil;
	XCTAssertTrue([server startWithError:&error], @"%@", error);
	[self addTeardownBlock:^{
		[server stop];
	}];
	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:server.localBaseURL];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }];
	XCTAssertNil([client runInferenceForRequest:request error:&error]);
	XCTAssertEqualObjects(error.domain, NFKInferenceServerErrorDomain);
	XCTAssertEqual(error.code, NFKInferenceServerErrorUnauthorized);
	client.apiKey = @"sesamf";
	XCTAssertNil([client runInferenceForRequest:request error:&error]);
	XCTAssertEqual(error.code, NFKInferenceServerErrorUnauthorized, @"a key of the right length but wrong is refused");
	client.apiKey = @"sesame";
	XCTAssertEqualObjects([client runInferenceForRequest:request error:&error].text, @"ok", @"%@", error);

	server.requiresAPIKeyOnLoopback = NO;
	client.apiKey = nil;
	XCTAssertEqualObjects([client runInferenceForRequest:request error:&error].text, @"ok", @"loopback is exempt by default: %@", error);
}

- (void)testALoopbackOnlyServerIsNotReachableFromANetworkAddress
{
	NSString *address = nil;
	struct ifaddrs *interfaces = NULL;
	if (getifaddrs(&interfaces) == 0) {
		for (struct ifaddrs *entry = interfaces; entry != NULL && address == nil; entry = entry->ifa_next) {
			if (entry->ifa_addr != NULL && entry->ifa_addr->sa_family == AF_INET && (entry->ifa_flags & IFF_LOOPBACK) == 0 && (entry->ifa_flags & IFF_UP)) {
				char text[INET_ADDRSTRLEN];
				inet_ntop(AF_INET, &((struct sockaddr_in *)(void *)entry->ifa_addr)->sin_addr, text, sizeof(text));
				address = @(text);
			}
		}
		freeifaddrs(interfaces);
	}
	if (address == nil) {
		XCTSkip(@"this machine has no non-loopback IPv4 address to connect from");
	}
	NFKInferenceServer *server = [self startedServerHosting:@{ @"local": [[NFKServedTestBackend alloc] init] }];
	int descriptor = socket(AF_INET, SOCK_STREAM, 0);
	struct sockaddr_in target = { .sin_len = sizeof(target), .sin_family = AF_INET, .sin_port = htons(server.listeningPort) };
	inet_pton(AF_INET, address.UTF8String, &target.sin_addr);
	int connected = connect(descriptor, (struct sockaddr *)&target, sizeof(target));
	close(descriptor);
	XCTAssertNotEqual(connected, 0, @"%@:%u accepted a connection", address, server.listeningPort);
}

#pragma mark Scheduling

- (void)testClosingTheConnectionCancelsTheHostedRun
{
	NFKServedHangingBackend *backend = [[NFKServedHangingBackend alloc] init];
	backend.cancelled = [self expectationWithDescription:@"the hosted job was cancelled"];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"slow": backend }];
	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:server.localBaseURL];
	NFKInferenceJob *job = [client submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }]];
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:5];
	while (!backend.started && deadline.timeIntervalSinceNow > 0) {
		[NSThread sleepForTimeInterval:0.02];
	}
	XCTAssertTrue(backend.started);
	[job cancel];
	[self waitForExpectations:@[ backend.cancelled ] timeout:5];
}

- (void)testOneModelServesOneRunAtATime
{
	NFKServedTestBackend *backend = [[NFKServedTestBackend alloc] init];
	backend.delay = 0.2;
	NFKInferenceServer *server = [self startedServerHosting:@{ @"serial": backend }];
	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:server.localBaseURL];
	dispatch_group_t group = dispatch_group_create();
	__block NSInteger succeeded = 0;
	for (NSInteger index = 0; index < 3; index++) {
		dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			NFKInferenceResult *result = [client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }] error:NULL];
			@synchronized (self) {
				succeeded += result != nil ? 1 : 0;
			}
		});
	}
	XCTAssertEqual(dispatch_group_wait(group, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(10 * NSEC_PER_SEC))), 0);
	XCTAssertEqual(succeeded, 3);
	XCTAssertEqual(backend.mostRunning, 1, @"the runs queued instead of overlapping");
}

#pragma mark HTTP

- (void)testAChunkedBodyAndUnknownRoutesAreHandled
{
	NFKServedTestBackend *backend = [[NFKServedTestBackend alloc] init];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"echo": backend }];
	NSData *JSON = [NSJSONSerialization dataWithJSONObject:@{ @"messages": @[ @{ @"role": @"user", @"content": @"streamed body" } ] } options:0 error:NULL];
	NSMutableURLRequest *chunked = [NSMutableURLRequest requestWithURL:[server.localBaseURL URLByAppendingPathComponent:@"chat/completions"]];
	chunked.HTTPMethod = @"POST";
	chunked.HTTPBodyStream = [NSInputStream inputStreamWithData:JSON];
	[chunked setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	NSHTTPURLResponse *response = nil;
	NSData *data = [self send:chunked response:&response];
	XCTAssertEqual(response.statusCode, 200, @"%@", data != nil ? [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] : @"");
	XCTAssertEqualObjects(backend.lastRequest.prompt, @"streamed body");

	[self send:[NSURLRequest requestWithURL:[server.localBaseURL URLByAppendingPathComponent:@"nowhere"]] response:&response];
	XCTAssertEqual(response.statusCode, 404);
	[self send:[NSURLRequest requestWithURL:[server.localBaseURL URLByAppendingPathComponent:@"chat/completions"]] response:&response];
	XCTAssertEqual(response.statusCode, 405);
	XCTAssertEqualObjects([response valueForHTTPHeaderField:@"Allow"], @"POST");
}

#pragma mark Status

- (nullable NSDictionary *)statusOf:(NFKInferenceServer *)server
{
	NSHTTPURLResponse *response = nil;
	NSData *data = [self send:[NSURLRequest requestWithURL:[server.localBaseURL URLByAppendingPathComponent:@"inferkit/status"]] response:&response];
	XCTAssertEqual(response.statusCode, 200);
	return data != nil ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
}

/*! The first model's load once it satisfies a condition, polling for up to five seconds. */
- (nullable NSDictionary *)loadOf:(NFKInferenceServer *)server when:(BOOL (^)(NSDictionary *load))condition
{
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:5];
	NSDictionary *load = nil;
	do {
		load = [[self statusOf:server][@"models"] firstObject][@"load"];
		if (load != nil && condition(load)) {
			return load;
		}
		[NSThread sleepForTimeInterval:0.02];
	} while (deadline.timeIntervalSinceNow > 0);
	return load;
}

- (NSMutableURLRequest *)chatRequestTo:(NFKInferenceServer *)server
{
	return [self postTo:[server.localBaseURL URLByAppendingPathComponent:@"chat/completions"]
				   JSON:@{ @"messages": @[ @{ @"role": @"user", @"content": @"x" } ] }];
}

- (void)testTheStatusRouteReportsTheModelsLoadAndTheMachine
{
	NFKServedTestBackend *backend = [[NFKServedTestBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"ok", NFKOutputUsage: @{ NFKUsageOutputTokens: @12 } }];
	};
	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.port = 0;
	server.loopbackOnly = YES;
	server.storageDirectoryURL = [NSURL fileURLWithPath:NSTemporaryDirectory()];
	[server addBackend:backend forModelName:@"measured"];
	NSError *error = nil;
	XCTAssertTrue([server startWithError:&error], @"%@", error);
	[self addTeardownBlock:^{
		[server stop];
	}];
	NSHTTPURLResponse *response = nil;
	[self send:[self chatRequestTo:server] response:&response];
	XCTAssertEqual(response.statusCode, 200);

	NSDictionary *status = [self statusOf:server];
	XCTAssertEqualObjects(status[@"object"], @"inferkit.status");
	XCTAssertEqualObjects(status[@"server"][@"version"], NFKInferKit.version);
	NSDictionary *model = [status[@"models"] firstObject];
	XCTAssertEqualObjects(model[@"id"], @"measured");
	XCTAssertEqualObjects(model[@"backend"], @"served-test");
	NSDictionary *load = model[@"load"];
	XCTAssertEqualObjects(load[@"limit"], @1);
	XCTAssertEqualObjects(load[@"running"], @0);
	XCTAssertEqualObjects(load[@"queued"], @0);
	XCTAssertEqualObjects(load[@"completed"], @1);
	XCTAssertEqualObjects(load[@"failed"], @0);
	XCTAssertEqualObjects(load[@"estimated_wait_seconds"], @0, @"a free slot starts a new run at once");
	XCTAssertGreaterThanOrEqual([load[@"average_run_seconds"] doubleValue], 0.0);
	XCTAssertGreaterThan([load[@"output_tokens_per_second"] doubleValue], 0.0);
	XCTAssertNil(load[@"queue_limit"]);

	NSDictionary *host = status[@"host"];
	NSArray *thermalStates = @[ @"nominal", @"fair", @"serious", @"critical" ];
	NSArray *pressures = @[ @"normal", @"warning", @"critical" ];
	XCTAssertTrue([thermalStates containsObject:host[@"thermal_state"]], @"%@", host[@"thermal_state"]);
	XCTAssertEqualObjects(host[@"chip"], NFKHardwareProfile.currentProfile.chipName);
	XCTAssertEqualObjects(host[@"memory"][@"physical_bytes"], @(NFKHardwareProfile.currentProfile.physicalMemory));
	XCTAssertTrue([pressures containsObject:host[@"memory"][@"pressure"]], @"%@", host[@"memory"][@"pressure"]);
	XCTAssertGreaterThan([host[@"memory"][@"process_footprint_bytes"] longLongValue], 0);
	XCTAssertEqual([host[@"cpu"][@"load_average"] count], 3u);
	double usage = [host[@"cpu"][@"usage"] doubleValue];
	XCTAssertTrue(usage >= 0.0 && usage <= 1.0, @"%f", usage);
	XCTAssertGreaterThan([host[@"storage"][@"available_bytes"] longLongValue], 0);
	NSDictionary *gpu = host[@"gpu"];
	if (gpu != nil) {
		XCTAssertEqualObjects(gpu[@"source"], @"ioregistry");
		XCTAssertTrue([gpu[@"utilization"] doubleValue] >= 0.0 && [gpu[@"utilization"] doubleValue] <= 1.0);
	}

	NSData *data = [self send:[NSURLRequest requestWithURL:[server.localBaseURL URLByAppendingPathComponent:@"models/measured"]] response:&response];
	NSDictionary *entry = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	XCTAssertEqualObjects(entry[@"inferkit"][@"load"][@"completed"], @1, @"the model entry carries the same load");
}

- (void)testRunRepliesCarryTheModelsLoadAndRefusalsDoNot
{
	NFKInferenceServer *server = [self startedServerHosting:@{ @"echo": [[NFKServedTestBackend alloc] init] }];
	NSHTTPURLResponse *response = nil;
	[self send:[self chatRequestTo:server] response:&response];
	XCTAssertEqual(response.statusCode, 200);
	XCTAssertEqualObjects([response valueForHTTPHeaderField:@"X-InferKit-Limit"], @"1");
	XCTAssertEqualObjects([response valueForHTTPHeaderField:@"X-InferKit-Running"], @"0", @"the head is written after the run released its slot");
	XCTAssertEqualObjects([response valueForHTTPHeaderField:@"X-InferKit-Queued"], @"0");
	XCTAssertEqualObjects([response valueForHTTPHeaderField:@"X-InferKit-Estimated-Wait"], @"0.000");
	XCTAssertNotNil([response valueForHTTPHeaderField:@"X-InferKit-Thermal-State"]);

	server.requiresAPIKeyOnLoopback = YES;
	server.apiKey = @"sesame";
	[self send:[self chatRequestTo:server] response:&response];
	XCTAssertEqual(response.statusCode, 401);
	XCTAssertNil([response valueForHTTPHeaderField:@"X-InferKit-Limit"]);
	XCTAssertNil([response valueForHTTPHeaderField:@"X-InferKit-Thermal-State"]);
	[self send:[NSURLRequest requestWithURL:[server.localBaseURL URLByAppendingPathComponent:@"inferkit/status"]] response:&response];
	XCTAssertEqual(response.statusCode, 401, @"the status route needs the key like every other route");
}

- (void)testHostDetailsCanBeLeftOutEntirely
{
	NFKInferenceServer *server = [self startedServerHosting:@{ @"private": [[NFKServedTestBackend alloc] init] }];
	server.reportsHostDetails = NO;
	NSHTTPURLResponse *response = nil;
	[self send:[self chatRequestTo:server] response:&response];
	XCTAssertEqual(response.statusCode, 200);
	XCTAssertNil([response valueForHTTPHeaderField:@"X-InferKit-Thermal-State"]);
	XCTAssertEqualObjects([response valueForHTTPHeaderField:@"X-InferKit-Limit"], @"1", @"the model's load stays");
	NSDictionary *status = [self statusOf:server];
	XCTAssertNil(status[@"host"]);
	XCTAssertEqualObjects([status[@"models"] firstObject][@"load"][@"completed"], @1);
}

- (void)testAFullQueueIsRefusedWithAnEstimatedRetry
{
	dispatch_semaphore_t gate = dispatch_semaphore_create(0);
	NFKServedTestBackend *backend = [[NFKServedTestBackend alloc] init];
	backend.reply = ^NFKInferenceResult *(NFKInferenceRequest *request, NSError **error) {
		dispatch_semaphore_wait(gate, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(10 * NSEC_PER_SEC)));
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"ok" }];
	};
	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.port = 0;
	server.loopbackOnly = YES;
	server.maximumQueuedRunsPerModel = 1;
	[server addBackend:backend forModelName:@"gated"];
	NSError *error = nil;
	XCTAssertTrue([server startWithError:&error], @"%@", error);
	[self addTeardownBlock:^{
		[server stop];
	}];
	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:server.localBaseURL];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }];
	dispatch_semaphore_signal(gate);
	XCTAssertEqualObjects([client runInferenceForRequest:request error:&error].text, @"ok", @"%@", error);

	dispatch_group_t group = dispatch_group_create();
	for (NSInteger index = 0; index < 2; index++) {
		dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			[client runInferenceForRequest:request error:NULL];
		});
	}
	NSDictionary *load = [self loadOf:server when:^BOOL(NSDictionary *candidate) {
		return [candidate[@"running"] integerValue] == 1 && [candidate[@"queued"] integerValue] == 1;
	}];
	XCTAssertEqualObjects(load[@"running"], @1);
	XCTAssertEqualObjects(load[@"queued"], @1);
	XCTAssertEqualObjects(load[@"queue_limit"], @1);
	XCTAssertNotNil(load[@"estimated_wait_seconds"], @"one finished run gives the queue an estimate");
	XCTAssertEqual([load[@"runs"] count], 1u);
	XCTAssertNotNil([load[@"runs"] firstObject][@"elapsed_seconds"]);

	NSHTTPURLResponse *response = nil;
	NSData *data = [self send:[self chatRequestTo:server] response:&response];
	XCTAssertEqual(response.statusCode, 503);
	XCTAssertGreaterThanOrEqual([[response valueForHTTPHeaderField:@"Retry-After"] integerValue], 1);
	XCTAssertEqualObjects([response valueForHTTPHeaderField:@"X-InferKit-Queued"], @"1");
	NSDictionary *body = data != nil ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
	XCTAssertEqualObjects(body[@"error"][@"inferkit_domain"], NFKInferenceServerErrorDomain);
	XCTAssertEqualObjects(body[@"error"][@"inferkit_code"], @(NFKInferenceServerErrorBusy));
	XCTAssertNil([client runInferenceForRequest:request error:&error]);
	XCTAssertEqualObjects(error.domain, NFKInferenceServerErrorDomain);
	XCTAssertEqual(error.code, NFKInferenceServerErrorBusy);

	dispatch_semaphore_signal(gate);
	dispatch_semaphore_signal(gate);
	XCTAssertEqual(dispatch_group_wait(group, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(10 * NSEC_PER_SEC))), 0);
	load = [self loadOf:server when:^BOOL(NSDictionary *candidate) {
		return [candidate[@"completed"] integerValue] == 3;
	}];
	XCTAssertEqualObjects(load[@"completed"], @3);
	XCTAssertEqualObjects(load[@"refused"], @2);
}

- (void)testEachModelEntryCarriesWhatItsBackendReportsAboutItsModel
{
	NFKInferenceServer *server = [self startedServerHosting:@{ @"described": [[NFKServedDescribedBackend alloc] init],
															   @"plain": [[NFKServedTestBackend alloc] init] }];
	NSDictionary *status = [self statusOf:server];
	NSDictionary *described = [status[@"models"] firstObject];
	NSDictionary *plain = [status[@"models"] lastObject];
	XCTAssertEqualObjects(described[@"id"], @"described");
	NSDictionary *expected = @{ NFKModelInfoParameterCount: @7000000, NFKModelInfoPrecision: @"bfloat16" };
	XCTAssertEqualObjects(described[@"model"], expected, @"a date and a non-finite number are left out");
	XCTAssertEqualObjects(plain[@"model"], @{}, @"a backend without modelInfo reports nothing");

	NSHTTPURLResponse *response = nil;
	NSData *data = [self send:[NSURLRequest requestWithURL:[server.localBaseURL URLByAppendingPathComponent:@"models/described"]] response:&response];
	NSDictionary *entry = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	XCTAssertEqualObjects(entry[@"inferkit"][@"model"], expected);
}

- (void)testALinkedRuntimeReportsUnderTheHostAndLeavesWithIt
{
	[NFKInferenceServer registerRuntimeStatusClassName:@"NFKServedTestRuntime"];
	[NFKInferenceServer registerRuntimeStatusClassName:@"NFKServedNoSuchRuntime"];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"echo": [[NFKServedTestBackend alloc] init] }];
	NSDictionary *runtimes = [self statusOf:server][@"host"][@"runtimes"];
	XCTAssertEqualObjects(runtimes[@"test-runtime"], @{ @"active_memory_bytes": @4096 });
	XCTAssertNil(runtimes[@"mlx"], @"InferKitMLX is not linked into the core's tests");
	server.reportsHostDetails = NO;
	XCTAssertNil([self statusOf:server][@"host"]);
}

- (void)testAClientReadsTheStatusTyped
{
	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.port = 0;
	server.loopbackOnly = YES;
	server.requiresAPIKeyOnLoopback = YES;
	server.apiKey = @"sesame";
	[server addBackend:[[NFKServedDescribedBackend alloc] init] forModelName:@"described"];
	NSError *error = nil;
	XCTAssertTrue([server startWithError:&error], @"%@", error);
	[self addTeardownBlock:^{
		[server stop];
	}];
	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:server.localBaseURL];
	client.apiKey = @"sesame";
	XCTAssertEqualObjects([client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }] error:&error].text,
						  @"ok", @"%@", error);

	NFKServerStatus *status = [client fetchServerStatusWithError:&error];
	XCTAssertNotNil(status, @"%@", error);
	XCTAssertEqualObjects(status.version, NFKInferKit.version);
	NFKServerModelStatus *model = [status modelNamed:@"described"];
	XCTAssertEqual(model.completed, 1);
	XCTAssertEqual(model.limit, 1);
	XCTAssertEqualObjects(model.estimatedWaitSeconds, @0);
	XCTAssertEqualObjects(model.modelInfo[NFKModelInfoPrecision], @"bfloat16");
	XCTAssertNotEqual(status.host.thermalState, NFKServerThermalStateUnknown);
	XCTAssertNotEqual(status.host.memoryPressure, NFKServerMemoryPressureUnknown);
	XCTAssertEqual(status.host.physicalMemory, NFKHardwareProfile.currentProfile.physicalMemory);

	XCTAssertNil([NFKServerStatus fetchFromBaseURL:server.localBaseURL apiKey:@"wrong" error:&error]);
	XCTAssertEqualObjects(error.domain, NFKInferenceServerErrorDomain);
	XCTAssertEqual(error.code, NFKInferenceServerErrorUnauthorized);

	XCTestExpectation *fetched = [self expectationWithDescription:@"the status arrives"];
	[NFKServerStatus fetchFromBaseURL:server.localBaseURL apiKey:@"sesame" completionHandler:^(NFKServerStatus *fetchedStatus, NSError *fetchError) {
		XCTAssertEqual([fetchedStatus modelNamed:@"described"].completed, 1, @"%@", fetchError);
		[fetched fulfill];
	}];
	[self waitForExpectations:@[ fetched ] timeout:5];
}

- (void)testARunningJobsProgressEstimatesItsRemainingTime
{
	NFKServedProgressingBackend *backend = [[NFKServedProgressingBackend alloc] init];
	backend.cancelled = [self expectationWithDescription:@"the hosted job was cancelled"];
	NFKInferenceServer *server = [self startedServerHosting:@{ @"progressing": backend }];
	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:server.localBaseURL];
	NFKInferenceJob *job = [client submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }]];
	NSDictionary *load = [self loadOf:server when:^BOOL(NSDictionary *candidate) {
		return [[candidate[@"runs"] firstObject][@"progress"] doubleValue] > 0;
	}];
	NSDictionary *run = [load[@"runs"] firstObject];
	XCTAssertEqualObjects(run[@"progress"], @0.25);
	double elapsed = [run[@"elapsed_seconds"] doubleValue];
	XCTAssertEqualWithAccuracy([run[@"estimated_remaining_seconds"] doubleValue], 3.0 * elapsed, 1e-9,
							   @"a quarter done after the elapsed time leaves three times it");
	XCTAssertEqualWithAccuracy([load[@"estimated_wait_seconds"] doubleValue], 3.0 * elapsed, 1e-9,
							   @"the next run waits for the one slot to free");
	[job cancel];
	[self waitForExpectations:@[ backend.cancelled ] timeout:5];
	load = [self loadOf:server when:^BOOL(NSDictionary *candidate) {
		return [candidate[@"cancelled"] integerValue] == 1;
	}];
	XCTAssertEqualObjects(load[@"cancelled"], @1);
	XCTAssertEqualObjects(load[@"running"], @0);
}

#pragma mark Discovery

- (void)testDiscoveryFindsAnAdvertisingServer
{
	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.port = 0;
	server.apiKey = @"discoverable";
	server.serviceName = [@"InferKit test " stringByAppendingString:[NSUUID.UUID.UUIDString substringToIndex:8]];
	[server addBackend:[[NFKServedTestBackend alloc] init] forModelName:@"found"];
	NSError *error = nil;
	XCTAssertTrue([server startWithError:&error], @"%@", error);
	[self addTeardownBlock:^{
		[server stop];
	}];
	NFKRemoteProvider *found = nil;
	for (NFKRemoteProvider *provider in [NFKRemoteProvider discoverInferKitServersWithTimeout:3.0]) {
		if ([provider.displayName isEqualToString:server.serviceName]) {
			found = provider;
		}
	}
	XCTAssertNotNil(found, @"the advertisement was not found over Bonjour");
	XCTAssertEqualObjects(found.identifier, @"inferkit");
	XCTAssertEqual(found.apiStyle, NFKRemoteAPIStyleInferKit);
	XCTAssertEqualObjects(found.baseURL.port, @(server.listeningPort));
	XCTAssertEqualObjects(found.baseURL.path, @"/v1");
	XCTAssertTrue(found.requiresAPIKey);
	XCTAssertEqualObjects(found.advertisedProperties[@"chip"], NFKHardwareProfile.currentProfile.chipName);
	XCTAssertEqualObjects(found.advertisedProperties[@"memory"],
						  ([NSString stringWithFormat:@"%ld", (long)NFKHardwareProfile.currentProfile.physicalMemory]));
	XCTAssertEqualObjects(found.advertisedProperties[@"version"], NFKInferKit.version);
	XCTAssertEqualObjects(NFKRemoteProvider.inferKit.advertisedProperties, @{}, @"a preset advertised nothing");

	// A client on another machine uses the advertised host name; from here it resolves to this machine.
	NFKRemoteInferKitBackend *client = (NFKRemoteInferKitBackend *)[NFKRemoteProvider backendForProvider:found apiKey:@"discoverable" modelName:nil];
	XCTAssertEqualObjects([client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }] error:&error].text,
						  @"ok", @"%@", error);
}

@end
