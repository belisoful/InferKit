//
//  NFKInferKitCTests.m
//  InferKitTests
//
//  The C interface of NFKInferKitC.h, called as a C program calls it: catalog backends built by
//  identifier with options, runs with JSON and with buffers of each kind, submitted runs with progress,
//  cancellation, and the ordering of frees, and programs in C99 and C++11 built and loaded at test time.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
#import <CoreVideo/CoreVideo.h>
#import <IOSurface/IOSurfaceRef.h>
#import <dlfcn.h>
#import <pthread.h>
#import <stdatomic.h>

#if __has_include(<InferKitC/NFKInferKitC.h>)
#import <InferKitC/NFKInferKitC.h>
#else
#import <InferKit/NFKInferKitC.h>
#endif

#if TARGET_OS_OSX
#import "NFKPluginFixtures.h"
#endif

static NSString * const NFKCAPITestEngine = @"c-api-test";
static NSString * const NFKCAPIEchoModel = @"c-api.echo";
static NSString * const NFKCAPIStreamingModel = @"c-api.streaming";
static NSString * const NFKCAPIUnavailableModel = @"c-api.unavailable";
static NSString * const NFKCAPIHugeModel = @"c-api.huge";

static const NSTimeInterval NFKCAPIDeadline = 30.0;

/*! A request whose run reads a 4096-byte data buffer 400 ms after it starts. */
static const char *NFKCAPISlowDataRequest =
	"{\"inputs\": {\"data\": {\"$nfk\": \"buffer\", \"index\": 0, \"as\": \"data\", \"length\": 4096}}, "
	"\"parameters\": {\"sleepMilliseconds\": 400, \"signalStart\": true}}";

static atomic_int NFKCAPIEchoRunsEnded;
static atomic_ullong NFKCAPIEchoLastDataSum;
static dispatch_semaphore_t NFKCAPIEchoRunStarted;

#pragma mark - Backends

/*! Echoes what it is given in a form a test checks: a prompt as "<mode>:<prompt>[:<suffix>]", a float
	array inverted to 1 - x, a 32BGRA image with each color byte inverted, and a data value's address,
	length, and byte sum, and with echoData the data itself under "echo". sleepMilliseconds holds the run
	before it reads the data. valueObjects adds one
	output of each InferKit value class; fail ends the run with an error that carries a cause. */
@interface NFKCAPIEchoBackend : NSObject <NFKInferenceBackend>
@property (nonatomic, copy) NSString *mode;
@property (nonatomic) BOOL failsPreparation;
@property (nonatomic) BOOL prepared;
@end

@implementation NFKCAPIEchoBackend

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_mode = @"plain";
	}
	return self;
}

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"c-api-echo";
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithObjects:NFKInputPrompt, NFKInputImage, @"values", @"data", @"region", nil];
}

- (NSSet<NSString *> *)supportedParameterKeys
{
	return [NSSet setWithObjects:@"suffix", @"sleepMilliseconds", @"signalStart", @"valueObjects", @"fail", @"echoData", nil];
}

- (BOOL)prepareWithError:(NSError **)error
{
	if (self.failsPreparation) {
		if (error != NULL) {
			*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceNotReady
									 userInfo:@{ NSLocalizedDescriptionKey: @"the echo model is set to fail loading" }];
		}
		return NO;
	}
	self.prepared = YES;
	return YES;
}

- (BOOL)isLoaded
{
	return self.prepared;
}

/*! One value of each InferKit value class, under the keys a backend returns them in. */
+ (NSDictionary<NSString *, id> *)valueObjectOutputs
{
	NFKQuadrilateral *corners = [NFKQuadrilateral quadrilateralWithTopLeft:CGPointMake(0.25, 0.75) topRight:CGPointMake(0.75, 0.75)
																bottomLeft:CGPointMake(0.25, 0.25) bottomRight:CGPointMake(0.75, 0.25)];
	NFKMIDINote *note = [NFKMIDINote noteWithPitch:60 startSeconds:0.5 endSeconds:1 velocity:90];
	return @{
		NFKOutputDetections: @[ [[NFKDetection alloc] initWithLabel:@"cat" classIndex:3 confidence:0.875
														 boundingBox:CGRectMake(0.25, 0.25, 0.5, 0.5) quadrilateral:corners] ],
		NFKOutputPose: @[ @[ [NFKKeypoint keypointWithName:@"nose" index:0 position:CGPointMake(0.5, 0.5) confidence:1] ] ],
		NFKOutputClassifications: @[ [NFKClassification classificationWithLabel:@"dog" classIndex:2 confidence:0.5] ],
		NFKOutputSegments: @[ [NFKAudioSegment segmentWithStartSeconds:0 endSeconds:1.5 label:@"hi" confidence:0.5] ],
		NFKOutputMIDI: [NFKMIDISequence sequenceWithNotes:@[ note ]],
		NFKOutputBeats: @[ [NFKMusicBeat beatWithTimeSeconds:0.5 positionInBar:1] ],
		NFKOutputAnswers: @{ @"spam": [[NFKDecisionAnswer alloc] initWithType:NFKDecisionTypeChoice choice:@"yes" score:0
																  probability:0 confidence:0.5 probabilities:nil legend:nil
																		  raw:@{ @"type": @"choice" }] },
	};
}

- (NSSet<NSString *> *)supportedOutputKeys
{
	return [NSSet setWithObjects:NFKOutputText, NFKOutputImage, @"values", @"dataAddress", @"dataLength", @"dataSum", @"regionLabel", @"echo", nil];
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	if ([[request parameterForKey:@"signalStart"] boolValue]) {
		dispatch_semaphore_signal(NFKCAPIEchoRunStarted);
	}
	NSInteger sleepMilliseconds = [[request parameterForKey:@"sleepMilliseconds"] integerValue];
	if (sleepMilliseconds > 0) {
		usleep((useconds_t)(sleepMilliseconds * 1000));
	}
	if ([[request parameterForKey:@"fail"] boolValue]) {
		NSError *cause = [NSError errorWithDomain:@"com.example.cause" code:7 userInfo:@{ NSLocalizedDescriptionKey: @"the cause" }];
		if (error != NULL) {
			*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceBackendFailure
									 userInfo:@{ NSLocalizedDescriptionKey: @"the echo model failed", NSUnderlyingErrorKey: cause,
												 @"note": @"kept", @"opaque": [[NSObject alloc] init] }];
		}
		return nil;
	}
	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionary];
	if ([[request parameterForKey:@"valueObjects"] boolValue]) {
		[outputs addEntriesFromDictionary:[NFKCAPIEchoBackend valueObjectOutputs]];
	}
	id region = [request inputForKey:@"region"];
	if ([region isKindOfClass:NFKDetection.class]) {
		outputs[@"regionLabel"] = [(NFKDetection *)region label];
	}
	NSString *prompt = [request inputForKey:NFKInputPrompt];
	if (prompt != nil) {
		NSString *suffix = [request parameterForKey:@"suffix"];
		outputs[NFKOutputText] = suffix != nil ? [NSString stringWithFormat:@"%@:%@:%@", self.mode, prompt, suffix]
											   : [NSString stringWithFormat:@"%@:%@", self.mode, prompt];
	}
	NSArray<NSNumber *> *values = [request inputForKey:@"values"];
	if (values != nil) {
		NSMutableArray<NSNumber *> *inverted = [NSMutableArray arrayWithCapacity:values.count];
		for (NSNumber *value in values) {
			[inverted addObject:@(1.0f - value.floatValue)];
		}
		outputs[@"values"] = inverted;
	}
	NSData *data = [request inputForKey:@"data"];
	if (data != nil) {
		const uint8_t *bytes = data.bytes;
		unsigned long long sum = 0;
		for (NSUInteger index = 0; index < data.length; index++) {
			sum += bytes[index];
		}
		outputs[@"dataAddress"] = @((unsigned long long)(uintptr_t)bytes);
		outputs[@"dataLength"] = @(data.length);
		outputs[@"dataSum"] = @(sum);
		atomic_store(&NFKCAPIEchoLastDataSum, sum);
		if ([[request parameterForKey:@"echoData"] boolValue]) {
			outputs[@"echo"] = data;
		}
	}
	id image = [request inputForKey:NFKInputImage];
	if (image != nil) {
		id inverted = [self invertedImage:image];
		if (inverted == nil) {
			if (error != NULL) {
				*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceMissingInput
										 userInfo:@{ NSLocalizedDescriptionKey: @"the image is not a 32BGRA pixel buffer" }];
			}
			return nil;
		}
		outputs[NFKOutputImage] = inverted;
	}
	atomic_fetch_add(&NFKCAPIEchoRunsEnded, 1);
	return [NFKInferenceResult resultWithOutputs:outputs];
}

- (nullable id)invertedImage:(id)image
{
	if (CFGetTypeID((__bridge CFTypeRef)image) != CVPixelBufferGetTypeID()) {
		return nil;
	}
	CVPixelBufferRef source = (__bridge CVPixelBufferRef)image;
	if (CVPixelBufferGetPixelFormatType(source) != kCVPixelFormatType_32BGRA) {
		return nil;
	}
	size_t width = CVPixelBufferGetWidth(source);
	size_t height = CVPixelBufferGetHeight(source);
	CVPixelBufferRef target = NULL;
	if (CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, NULL, &target) != kCVReturnSuccess) {
		return nil;
	}
	CVPixelBufferLockBaseAddress(source, kCVPixelBufferLock_ReadOnly);
	CVPixelBufferLockBaseAddress(target, 0);
	const uint8_t *from = CVPixelBufferGetBaseAddress(source);
	uint8_t *to = CVPixelBufferGetBaseAddress(target);
	size_t fromRow = CVPixelBufferGetBytesPerRow(source);
	size_t toRow = CVPixelBufferGetBytesPerRow(target);
	for (size_t y = 0; y < height; y++) {
		for (size_t x = 0; x < width; x++) {
			const uint8_t *pixel = from + y * fromRow + x * 4;
			uint8_t *out = to + y * toRow + x * 4;
			out[0] = 255 - pixel[0];
			out[1] = 255 - pixel[1];
			out[2] = 255 - pixel[2];
			out[3] = pixel[3];
		}
	}
	CVPixelBufferUnlockBaseAddress(target, 0);
	CVPixelBufferUnlockBaseAddress(source, kCVPixelBufferLock_ReadOnly);
	return (__bridge_transfer id)target;
}

@end

/*! Counts "steps" steps (default 5) of "stepMilliseconds" each (default 10), reporting progress and a
	partial result {"step": n} after each, and ends cancelled once its job is cancelled. */
@interface NFKCAPIStreamingBackend : NSObject <NFKInferenceBackend>
@end

@implementation NFKCAPIStreamingBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"c-api-streaming";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"done" }];
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	NSInteger steps = [[request parameterForKey:@"steps"] integerValue] ?: 5;
	NSInteger stepMilliseconds = [[request parameterForKey:@"stepMilliseconds"] integerValue] ?: 10;
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		for (NSInteger step = 1; step <= steps; step++) {
			if (job.status == NFKInferenceJobStatusCancelled) {
				return;
			}
			usleep((useconds_t)(stepMilliseconds * 1000));
			NFKDetection *detection = [NFKDetection detectionWithLabel:@"step" classIndex:step confidence:1 boundingBox:CGRectMake(0, 0, 1, 1)];
			NFKInferenceResult *partial = [NFKInferenceResult resultWithOutputs:@{ @"step": @(step), NFKOutputDetections: @[ detection ] }];
			[job reportProgress:(double)step / (double)steps partialResult:partial];
		}
		[job finishWithResult:[NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"done" }]];
	});
	return job;
}

@end

/*! Builds the streaming backend for its model and the echo backend for the others. */
@interface NFKCAPITestBuilder : NSObject <NFKModelBackendBuilding>
@end

@implementation NFKCAPITestBuilder

- (nullable id<NFKInferenceBackend>)backendForBuildRequest:(NFKModelBuildRequest *)request error:(NSError **)error
{
	if ([request.descriptor.identifier isEqualToString:NFKCAPIStreamingModel]) {
		return [[NFKCAPIStreamingBackend alloc] init];
	}
	NFKCAPIEchoBackend *backend = [[NFKCAPIEchoBackend alloc] init];
	return [request applyPropertyOptionsToBackend:backend error:error] ? backend : nil;
}

@end

#pragma mark - Callback log

/*! What a submitted run's callbacks reported. The C callbacks receive it as their context. */
@interface NFKCAPICallbackLog : NSObject
@property (nonatomic, readonly) NSLock *lock;
@property (nonatomic, readonly) dispatch_semaphore_t completed;
@property (nonatomic, readonly) dispatch_semaphore_t progressed;
@property (nonatomic) NSInteger progressCount;
@property (nonatomic) NSInteger completionCount;
@property (nonatomic) NSInteger callbacksAfterFree;
@property (nonatomic) BOOL freed;
@property (nonatomic) BOOL overlapped;
@property (nonatomic) BOOL inCallback;
@property (nonatomic) BOOL ranOnSubmittingThread;
@property (nonatomic) BOOL ranBelowUserInitiated;
@property (nonatomic, nullable) NSString *lastPartialJSON;
@property (nonatomic, nullable) NSString *resultJSON;
@property (nonatomic, nullable) NSString *errorDomain;
@property (nonatomic) int64_t errorCode;
@property (nonatomic) int endedRunsAtCompletion;
@property (nonatomic) pthread_t submittingThread;
@property (nonatomic, nullable) nfk_job *jobToFreeInCompletion;
@property (nonatomic, nullable) nfk_job *jobToFreeInProgress;
@property (nonatomic) int endedRunsAtFree;
@end

@implementation NFKCAPICallbackLog

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_lock = [[NSLock alloc] init];
		_completed = dispatch_semaphore_create(0);
		_progressed = dispatch_semaphore_create(0);
		_submittingThread = pthread_self();
	}
	return self;
}

- (void)enterCallback
{
	[self.lock lock];
	self.overlapped = self.overlapped || self.inCallback;
	self.inCallback = YES;
	self.ranOnSubmittingThread = self.ranOnSubmittingThread || pthread_equal(pthread_self(), self.submittingThread);
	self.ranBelowUserInitiated = self.ranBelowUserInitiated || qos_class_self() < QOS_CLASS_USER_INITIATED;
	if (self.freed) {
		self.callbacksAfterFree += 1;
	}
	[self.lock unlock];
}

- (void)leaveCallback
{
	[self.lock lock];
	self.inCallback = NO;
	[self.lock unlock];
}

- (void)markFreed
{
	[self.lock lock];
	self.freed = YES;
	[self.lock unlock];
}

@end

static void NFKCAPITestProgress(void *context, double fraction, const nfk_result *partial)
{
	NFKCAPICallbackLog *log = (__bridge NFKCAPICallbackLog *)context;
	[log enterCallback];
	[log.lock lock];
	log.progressCount += 1;
	if (partial != NULL) {
		log.lastPartialJSON = @(nfk_result_json(partial));
	}
	nfk_job *job = log.jobToFreeInProgress;
	log.jobToFreeInProgress = NULL;
	[log.lock unlock];
	if (job != NULL) {
		nfk_job_free(job);
		[log.lock lock];
		log.endedRunsAtFree = atomic_load(&NFKCAPIEchoRunsEnded);
		[log.lock unlock];
	}
	[log leaveCallback];
	dispatch_semaphore_signal(log.progressed);
}

static void NFKCAPITestCompletion(void *context, nfk_result *result, nfk_error *error)
{
	NFKCAPICallbackLog *log = (__bridge NFKCAPICallbackLog *)context;
	[log enterCallback];
	[log.lock lock];
	log.completionCount += 1;
	log.endedRunsAtCompletion = atomic_load(&NFKCAPIEchoRunsEnded);
	log.resultJSON = result != NULL ? @(nfk_result_json(result)) : nil;
	log.errorDomain = error != NULL ? @(nfk_error_domain(error)) : nil;
	log.errorCode = nfk_error_code(error);
	nfk_job *job = log.jobToFreeInCompletion;
	[log.lock unlock];
	nfk_result_free(result);
	nfk_error_free(error);
	if (job != NULL) {
		nfk_job_free(job);
	}
	[log leaveCallback];
	dispatch_semaphore_signal(log.completed);
}

#pragma mark - Tests

@interface NFKInferKitCTests : XCTestCase
@property (nonatomic) NFKCAPITestBuilder *builder;
@end

@implementation NFKInferKitCTests

+ (void)setUp
{
	NFKCAPIEchoRunStarted = dispatch_semaphore_create(0);
}

- (void)setUp
{
	self.builder = [[NFKCAPITestBuilder alloc] init];
	[NFKModelCatalog registerBackendBuilder:self.builder forEngine:NFKCAPITestEngine];
	NFKModelCatalog *catalog = NFKModelCatalog.sharedCatalog;
	// A 64 GiB working set, so no build depends on the memory of the machine that runs the tests.
	catalog.memoryBudget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:(NSInteger)64 << 30];
	[catalog addDescriptor:[self echoDescriptor]];
	[catalog addDescriptor:[self descriptorWithIdentifier:NFKCAPIStreamingModel options:nil requirements:nil]];
	NFKModelRequirements *unmeetable = [[NFKModelRequirements alloc] initWithMinimumOSVersions:nil requiresAppleSilicon:NO
																		minimumMemoryByteCount:nil
																			  requiredFeatures:@[ @"made-up" ]
																				   unmetReason:nil extensions:nil];
	[catalog addDescriptor:[self descriptorWithIdentifier:NFKCAPIUnavailableModel options:nil requirements:unmeetable]];
	NFKModelVariant *huge = [[NFKModelVariant alloc] initWithIdentifier:@"huge" displayName:nil isDefault:YES registryName:nil source:nil
													  downloadByteCount:nil residentByteCount:@((NSInteger)1 << 50) peakByteCount:nil
														 parameterCount:nil precision:nil minimumMemoryByteCount:nil extensions:nil];
	[catalog addDescriptor:[[NFKModelDescriptor alloc] initWithIdentifier:NFKCAPIHugeModel displayName:@"C API Huge Model" family:@"test"
																  summary:nil engine:NFKCAPITestEngine registryName:nil
														providerClassName:nil tasks:nil inputModalities:nil outputModalities:nil
																   inputs:nil outputs:nil options:nil variants:@[ huge ] license:nil
															 requirements:nil residencies:nil standIn:NO introducedVersion:nil
															   extensions:nil]];
}

- (void)tearDown
{
	NFKModelCatalog *catalog = NFKModelCatalog.sharedCatalog;
	[catalog removeDescriptorWithIdentifier:NFKCAPIEchoModel];
	[catalog removeDescriptorWithIdentifier:NFKCAPIStreamingModel];
	[catalog removeDescriptorWithIdentifier:NFKCAPIUnavailableModel];
	[catalog removeDescriptorWithIdentifier:NFKCAPIHugeModel];
	catalog.memoryBudget = nil;
	[NFKModelCatalog registerBackendBuilder:nil forEngine:NFKCAPITestEngine];
}

- (NFKModelDescriptor *)descriptorWithIdentifier:(NSString *)identifier
										 options:(nullable NSArray<NFKModelOption *> *)options
									requirements:(nullable NFKModelRequirements *)requirements
{
	return [[NFKModelDescriptor alloc] initWithIdentifier:identifier displayName:@"C API Model" family:@"test" summary:nil
												   engine:NFKCAPITestEngine registryName:nil providerClassName:nil tasks:nil
										  inputModalities:nil outputModalities:nil inputs:nil outputs:nil options:options
												 variants:nil license:nil requirements:requirements residencies:nil standIn:NO
										introducedVersion:nil extensions:nil];
}

// A load option the builder sets on the backend's mode property, and a request option.
- (NFKModelDescriptor *)echoDescriptor
{
	NSArray<NFKModelOption *> *options = @[
		[[NFKModelOption alloc] initWithKey:@"mode" kind:NFKModelOptionKindText title:nil help:nil defaultValue:@"plain" minimum:nil
									maximum:nil step:nil choices:nil stage:NFKModelOptionStageLoad group:nil requirements:nil
							propertyKeyPath:@"mode" memoryBytesPerUnit:nil extensions:nil],
		[[NFKModelOption alloc] initWithKey:@"suffix" kind:NFKModelOptionKindText title:nil help:nil defaultValue:nil minimum:nil
									maximum:nil step:nil choices:nil stage:NFKModelOptionStageRequest group:nil extensions:nil],
		[[NFKModelOption alloc] initWithKey:@"failPrepare" kind:NFKModelOptionKindBoolean title:nil help:nil defaultValue:@NO minimum:nil
									maximum:nil step:nil choices:nil stage:NFKModelOptionStageLoad group:nil requirements:nil
							propertyKeyPath:@"failsPreparation" memoryBytesPerUnit:nil extensions:nil],
	];
	return [self descriptorWithIdentifier:NFKCAPIEchoModel options:options requirements:nil];
}

#pragma mark Helpers

- (BOOL)wait:(dispatch_semaphore_t)semaphore seconds:(NSTimeInterval)seconds
{
	return dispatch_semaphore_wait(semaphore, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(seconds * NSEC_PER_SEC))) == 0;
}

- (nfk_backend *)backendFor:(NSString *)identifier options:(nullable const char *)options
{
	nfk_error *error = NULL;
	nfk_backend *backend = nfk_backend_create(identifier.UTF8String, options, &error);
	XCTAssert(backend != NULL, @"%s", nfk_error_message(error));
	XCTAssert(error == NULL);
	nfk_error_free(error);
	return backend;
}

- (nfk_payload)payloadWithJSON:(const char *)json buffers:(nullable const nfk_buffer *)buffers count:(uint32_t)count
{
	nfk_payload payload;
	memset(&payload, 0, sizeof payload);
	payload.struct_size = (uint32_t)sizeof payload;
	payload.json = json;
	payload.buffers = buffers;
	payload.buffer_count = count;
	return payload;
}

- (nfk_buffer)bytesBuffer:(const void *)bytes length:(uint64_t)length
{
	nfk_buffer buffer;
	memset(&buffer, 0, sizeof buffer);
	buffer.struct_size = (uint32_t)sizeof buffer;
	buffer.kind = NFK_BUFFER_BYTES;
	buffer.bytes = bytes;
	buffer.length = length;
	buffer.descriptor = -1;
	return buffer;
}

- (nullable NSDictionary *)JSONObject:(nullable const char *)json
{
	if (json == NULL) {
		return nil;
	}
	NSData *data = [NSData dataWithBytes:json length:strlen(json)];
	id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	return [object isKindOfClass:NSDictionary.class] ? object : nil;
}

- (nullable NSDictionary *)outputsOfResult:(nullable const nfk_result *)result
{
	return [self JSONObject:nfk_result_json(result)][@"outputs"];
}

/*! A pixelBuffer attachment entry for a 32BGRA image of side by side pixels with rows of bytesPerRow. */
- (NSString *)imageEntryOfSide:(NSUInteger)side bytesPerRow:(NSUInteger)bytesPerRow
{
	return [NSString stringWithFormat:@"{\"$nfk\": \"buffer\", \"index\": 0, \"as\": \"pixelBuffer\", \"length\": %lu, "
			"\"format\": %u, \"width\": %lu, \"height\": %lu, \"planes\": [{\"offset\": 0, \"bytesPerRow\": %lu, \"rows\": %lu}]}",
			(unsigned long)(bytesPerRow * side), (unsigned)kCVPixelFormatType_32BGRA, (unsigned long)side, (unsigned long)side,
			(unsigned long)bytesPerRow, (unsigned long)side];
}

/*! Checks that a result's image is the inverse of source: each color byte 255 minus the source's. */
- (void)assertResult:(const nfk_result *)result invertsImage:(const uint8_t *)source side:(NSUInteger)side bytesPerRow:(NSUInteger)bytesPerRow
{
	NSDictionary *image = [self outputsOfResult:result][NFKOutputImage];
	XCTAssertEqualObjects(image[@"$nfk"], @"buffer");
	XCTAssertEqualObjects(image[@"as"], @"pixelBuffer");
	XCTAssertEqualObjects(image[@"width"], @(side));
	NSDictionary *plane = [image[@"planes"] firstObject];
	const nfk_buffer *buffer = nfk_result_buffer(result, [image[@"index"] unsignedIntValue]);
	XCTAssert(buffer != NULL);
	if (buffer == NULL || plane == nil) {
		return;
	}
	XCTAssertEqual(buffer->kind, NFK_BUFFER_BYTES);
	const uint8_t *bytes = (const uint8_t *)buffer->bytes + [plane[@"offset"] unsignedIntegerValue];
	NSUInteger outputRow = [plane[@"bytesPerRow"] unsignedIntegerValue];
	NSUInteger mismatches = 0;
	for (NSUInteger y = 0; y < side; y++) {
		for (NSUInteger x = 0; x < side; x++) {
			const uint8_t *in = source + y * bytesPerRow + x * 4;
			const uint8_t *out = bytes + y * outputRow + x * 4;
			BOOL inverted = out[0] == 255 - in[0] && out[1] == 255 - in[1] && out[2] == 255 - in[2] && out[3] == in[3];
			mismatches += inverted ? 0 : 1;
		}
	}
	XCTAssertEqual(mismatches, 0u);
}

#pragma mark Versions and the catalog

- (void)testTheVersionFunctionsReportTheLibrary
{
	XCTAssertEqual(nfk_api_version(), (uint32_t)NFK_API_VERSION);
	XCTAssertEqual(nfk_api_version() >> 16, (uint32_t)NFKInferKit.APIMajorVersion);
	XCTAssertEqual(nfk_api_version() & 0xFFFF, (uint32_t)NFKInferKit.APIMinorVersion);
	XCTAssertEqualObjects(@(nfk_release_version()), NFKInferKit.releaseVersion);
	XCTAssertEqual(nfk_release_version(), nfk_release_version(), @"the version lives in static storage");
}

- (void)testTheCatalogJSONIsTheSharedCatalogsManifest
{
	nfk_error *error = NULL;
	char *json = nfk_catalog_copy_json(&error);
	XCTAssert(json != NULL, @"%s", nfk_error_message(error));
	XCTAssert(error == NULL);
	NSDictionary *manifest = [self JSONObject:json];
	XCTAssertEqualObjects(manifest[@"kind"], NFKModelCatalogManifestKind);
	NSArray *identifiers = [manifest[@"models"] valueForKey:@"id"];
	XCTAssertTrue([identifiers containsObject:@"vision-feature-print"], @"%@", identifiers);
	XCTAssertTrue([identifiers containsObject:NFKCAPIEchoModel]);
	nfk_string_free(json);

	char *withoutError = nfk_catalog_copy_json(NULL);
	XCTAssert(withoutError != NULL);
	nfk_string_free(withoutError);
}

#pragma mark Building

- (void)testAnAppleModelBuildsByIdentifierAndEmbedsAnImageBuffer
{
	nfk_backend *backend = [self backendFor:@"vision-feature-print" options:NULL];
	char *description = nfk_backend_copy_description(backend);
	NSDictionary *described = [self JSONObject:description];
	XCTAssertEqualObjects(described[@"identifier"], @"vision-feature-print");
	XCTAssertTrue([described[@"inputs"] containsObject:NFKInputImage], @"%@", described);
	XCTAssertEqualObjects(described[@"ready"], @YES);
	nfk_string_free(description);

	const NSUInteger side = 128;
	NSMutableData *pixels = [NSMutableData dataWithLength:side * side * 4];
	uint8_t *bytes = pixels.mutableBytes;
	for (NSUInteger index = 0; index < pixels.length; index++) {
		bytes[index] = (uint8_t)((index * 7) % 251);
	}
	nfk_buffer buffer = [self bytesBuffer:bytes length:pixels.length];
	NSString *json = [NSString stringWithFormat:@"{\"inputs\": {\"image\": %@}}", [self imageEntryOfSide:side bytesPerRow:side * 4]];
	nfk_payload request = [self payloadWithJSON:json.UTF8String buffers:&buffer count:1];
	nfk_error *error = NULL;
	nfk_result *result = nfk_backend_run(backend, &request, &error);
	XCTAssert(result != NULL, @"%s", nfk_error_message(error));
	NSDictionary *embedding = [self outputsOfResult:result][NFKOutputEmbedding];
	XCTAssertEqualObjects(embedding[@"$nfk"], @"buffer", @"every value with bytes comes back as a buffer");
	const nfk_buffer *vector = nfk_result_buffer(result, [embedding[@"index"] unsignedIntValue]);
	XCTAssert(vector != NULL && vector->length > 0);
	nfk_result_free(result);
	nfk_error_free(error);
	nfk_backend_free(backend);
}

- (void)testOptionsReachTheBackendThroughTheCatalog
{
	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:"{\"mode\": \"loud\", \"suffix\": \"?\"}"];
	nfk_payload request = [self payloadWithJSON:"{\"inputs\": {\"prompt\": \"hi\"}, \"parameters\": {}, \"outputModality\": 0}"
										buffers:NULL count:0];
	nfk_error *error = NULL;
	nfk_result *result = nfk_backend_run(backend, &request, &error);
	XCTAssert(result != NULL, @"%s", nfk_error_message(error));
	XCTAssertEqualObjects([self outputsOfResult:result][NFKOutputText], @"loud:hi:?",
						  @"the load option sets the mode and the request option fills the suffix");
	XCTAssertEqual(nfk_result_buffer_count(result), 0u);
	nfk_result_free(result);
	nfk_backend_free(backend);
}

- (void)testAnUnknownModelFailsWithTheCatalogsDomainAndCode
{
	nfk_error *error = NULL;
	XCTAssert(nfk_backend_create("c-api.ghost", NULL, &error) == NULL);
	XCTAssertEqualObjects(@(nfk_error_domain(error)), NFKModelCatalogErrorDomain);
	XCTAssertEqual(nfk_error_code(error), (int64_t)NFKModelCatalogErrorUnknownModel);
	XCTAssertTrue([@(nfk_error_message(error)) containsString:@"c-api.ghost"], @"%s", nfk_error_message(error));
	nfk_error_free(error);

	error = NULL;
	XCTAssert(nfk_backend_create(NULL, NULL, &error) == NULL);
	XCTAssertEqual(nfk_error_code(error), (int64_t)NFKModelCatalogErrorUnknownModel);
	nfk_error_free(error);
}

- (void)testOptionsTheModelDoesNotTakeAreInvalidOptions
{
	const char *refused[] = { "{\"nope\": 1}", "{\"mode\":", "[1]", "\"loud\"" };
	for (size_t index = 0; index < sizeof refused / sizeof refused[0]; index++) {
		nfk_error *error = NULL;
		XCTAssert(nfk_backend_create(NFKCAPIEchoModel.UTF8String, refused[index], &error) == NULL, @"%s", refused[index]);
		XCTAssertEqualObjects(@(nfk_error_domain(error)), NFKModelCatalogErrorDomain, @"%s", refused[index]);
		XCTAssertEqual(nfk_error_code(error), (int64_t)NFKModelCatalogErrorInvalidOptions, @"%s", refused[index]);
		nfk_error_free(error);
	}
}

- (void)testAModelThisMachineCannotRunIsUnavailable
{
	nfk_error *error = NULL;
	XCTAssert(nfk_backend_create(NFKCAPIUnavailableModel.UTF8String, "", &error) == NULL);
	XCTAssertEqual(nfk_error_code(error), (int64_t)NFKModelCatalogErrorUnavailable);
	XCTAssertEqualObjects(@(nfk_error_message(error)), @"C API Model cannot run on this machine. This needs the made-up feature.");
	nfk_error_free(error);
}

#pragma mark Running with buffers

- (void)testAFloatBufferRunsAndComesBackAsAFloatBuffer
{
	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:NULL];
	float values[1024];
	for (NSUInteger index = 0; index < 1024; index++) {
		values[index] = (float)index / 1023.0f;
	}
	nfk_buffer buffer = [self bytesBuffer:values length:sizeof values];
	nfk_payload request = [self payloadWithJSON:"{\"inputs\": {\"values\": {\"$nfk\": \"buffer\", \"index\": 0, \"as\": \"float32Array\", "
										"\"length\": 4096, \"count\": 1024}}}"
										buffers:&buffer count:1];
	nfk_error *error = NULL;
	nfk_result *result = nfk_backend_run(backend, &request, &error);
	XCTAssert(result != NULL, @"%s", nfk_error_message(error));
	NSDictionary *entry = [self outputsOfResult:result][@"values"];
	XCTAssertEqualObjects(entry[@"as"], @"float32Array");
	XCTAssertEqualObjects(entry[@"count"], @1024);
	const nfk_buffer *output = nfk_result_buffer(result, [entry[@"index"] unsignedIntValue]);
	XCTAssert(output != NULL && output->length == sizeof values);
	NSUInteger mismatches = 0;
	for (NSUInteger index = 0; output != NULL && index < 1024; index++) {
		mismatches += ((const float *)output->bytes)[index] == 1.0f - values[index] ? 0 : 1;
	}
	XCTAssertEqual(mismatches, 0u);
	XCTAssert(nfk_result_buffer(result, nfk_result_buffer_count(result)) == NULL, @"past the end is NULL");
	nfk_result_free(result);
	nfk_backend_free(backend);
}

- (void)testStructuresAtTheOneZeroSizeAndLongerOnesAreRead
{
	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:NULL];
	uint8_t decoy[16];
	uint8_t data[64];
	memset(decoy, 9, sizeof decoy);
	memset(data, 2, sizeof data);
	const char *json = "{\"inputs\": {\"data\": {\"$nfk\": \"buffer\", \"index\": 1, \"as\": \"data\", \"length\": 64}}}";
	// A writer built against a later minor version: each buffer 16 bytes longer, the payload too.
	const size_t longer = 16;
	struct { nfk_payload payload; uint8_t later[16]; } longPayload;
	memset(&longPayload, 0, sizeof longPayload);
	_Alignas(nfk_buffer) uint8_t longBuffers[2 * (sizeof(nfk_buffer) + 16)];
	memset(longBuffers, 0, sizeof longBuffers);
	const void *sources[2] = { decoy, data };
	const uint64_t lengths[2] = { sizeof decoy, sizeof data };
	for (NSUInteger index = 0; index < 2; index++) {
		nfk_buffer buffer = [self bytesBuffer:sources[index] length:lengths[index]];
		buffer.struct_size = (uint32_t)(sizeof(nfk_buffer) + longer);
		memcpy(longBuffers + index * (sizeof(nfk_buffer) + longer), &buffer, sizeof buffer);
	}
	longPayload.payload.struct_size = (uint32_t)(sizeof(nfk_payload) + longer);
	longPayload.payload.json = json;
	longPayload.payload.buffers = (const nfk_buffer *)longBuffers;
	longPayload.payload.buffer_count = 2;
	// A writer built against 1.0 that sizes each structure through its last 1.0 field.
	nfk_buffer shortBuffers[2] = { [self bytesBuffer:decoy length:sizeof decoy], [self bytesBuffer:data length:sizeof data] };
	shortBuffers[0].struct_size = (uint32_t)(offsetof(nfk_buffer, reserved) + sizeof(uint32_t));
	shortBuffers[1].struct_size = shortBuffers[0].struct_size;
	nfk_payload shortPayload = [self payloadWithJSON:json buffers:shortBuffers count:2];
	shortPayload.struct_size = (uint32_t)(offsetof(nfk_payload, buffers) + sizeof(const nfk_buffer *));

	const nfk_payload *requests[2] = { &shortPayload, &longPayload.payload };
	for (NSUInteger index = 0; index < 2; index++) {
		nfk_error *error = NULL;
		nfk_result *result = nfk_backend_run(backend, requests[index], &error);
		XCTAssert(result != NULL, @"request %lu: %s", (unsigned long)index, nfk_error_message(error));
		XCTAssertEqualObjects([self outputsOfResult:result][@"dataSum"], @(2 * 64), @"request %lu strides to buffer 1", (unsigned long)index);
		nfk_result_free(result);
		nfk_error_free(error);
	}

	nfk_payload tooShort = shortPayload;
	tooShort.struct_size -= 1;
	nfk_error *error = NULL;
	XCTAssert(nfk_backend_run(backend, &tooShort, &error) == NULL, @"a payload shorter than 1.0's is refused");
	XCTAssertEqual(nfk_error_code(error), (int64_t)kNFKError_InferenceMissingInput);
	XCTAssertTrue(strstr(nfk_error_message(error) ?: "", "1.0") != NULL, @"%s", nfk_error_message(error));
	nfk_error_free(error);
	nfk_backend_free(backend);
}

- (void)testABytesBufferIsWrappedWithoutACopy
{
	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:NULL];
	NSMutableData *data = [NSMutableData dataWithLength:4096];
	memset(data.mutableBytes, 3, data.length);
	nfk_buffer buffer = [self bytesBuffer:data.bytes length:data.length];
	nfk_payload request = [self payloadWithJSON:"{\"inputs\": {\"data\": {\"$nfk\": \"buffer\", \"index\": 0, \"as\": \"data\", \"length\": 4096}}}"
										buffers:&buffer count:1];
	nfk_error *error = NULL;
	nfk_result *result = nfk_backend_run(backend, &request, &error);
	XCTAssert(result != NULL, @"%s", nfk_error_message(error));
	NSDictionary *outputs = [self outputsOfResult:result];
	XCTAssertEqualObjects(outputs[@"dataAddress"], @((unsigned long long)(uintptr_t)data.bytes), @"the backend read the caller's memory");
	XCTAssertEqualObjects(outputs[@"dataSum"], @(3 * 4096));
	nfk_result_free(result);
	nfk_backend_free(backend);
}

- (void)testAnImageBufferRunsAndComesBackInverted
{
	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:NULL];
	const NSUInteger side = 64;
	const NSUInteger bytesPerRow = 320;
	NSMutableData *pixels = [NSMutableData dataWithLength:bytesPerRow * side];
	uint8_t *bytes = pixels.mutableBytes;
	for (NSUInteger index = 0; index < pixels.length; index++) {
		bytes[index] = (uint8_t)(index % 253);
	}
	nfk_buffer buffer = [self bytesBuffer:bytes length:pixels.length];
	NSString *json = [NSString stringWithFormat:@"{\"inputs\": {\"image\": %@}}", [self imageEntryOfSide:side bytesPerRow:bytesPerRow]];
	nfk_payload request = [self payloadWithJSON:json.UTF8String buffers:&buffer count:1];
	nfk_error *error = NULL;
	nfk_result *result = nfk_backend_run(backend, &request, &error);
	XCTAssert(result != NULL, @"%s", nfk_error_message(error));
	if (result != NULL) {
		[self assertResult:result invertsImage:bytes side:side bytesPerRow:bytesPerRow];
	}
	nfk_result_free(result);
	nfk_backend_free(backend);
}

- (void)testASurfaceBufferRunsAndComesBackInverted
{
	const NSUInteger side = 64;
	NSDictionary *properties = @{ (id)kIOSurfaceWidth: @(side), (id)kIOSurfaceHeight: @(side), (id)kIOSurfaceBytesPerElement: @4,
								  (id)kIOSurfacePixelFormat: @(kCVPixelFormatType_32BGRA) };
	IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)properties);
	XCTAssert(surface != NULL);
	if (surface == NULL) {
		return;
	}
	NSUInteger bytesPerRow = IOSurfaceGetBytesPerRow(surface);
	IOSurfaceLock(surface, 0, NULL);
	uint8_t *bytes = IOSurfaceGetBaseAddress(surface);
	for (NSUInteger index = 0; index < bytesPerRow * side; index++) {
		bytes[index] = (uint8_t)(index % 241);
	}
	NSData *source = [NSData dataWithBytes:bytes length:bytesPerRow * side];
	IOSurfaceUnlock(surface, 0, NULL);

	nfk_buffer buffer;
	memset(&buffer, 0, sizeof buffer);
	buffer.struct_size = (uint32_t)sizeof buffer;
	buffer.kind = NFK_BUFFER_SURFACE;
	buffer.surface = surface;
	buffer.descriptor = -1;
	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:NULL];
	NSString *json = [NSString stringWithFormat:@"{\"inputs\": {\"image\": %@}}", [self imageEntryOfSide:side bytesPerRow:bytesPerRow]];
	nfk_payload request = [self payloadWithJSON:json.UTF8String buffers:&buffer count:1];
	nfk_error *error = NULL;
	nfk_result *result = nfk_backend_run(backend, &request, &error);
	XCTAssert(result != NULL, @"%s", nfk_error_message(error));
	if (result != NULL) {
		[self assertResult:result invertsImage:source.bytes side:side bytesPerRow:bytesPerRow];
	}
	nfk_result_free(result);
	nfk_backend_free(backend);
	CFRelease(surface);
}

- (void)testAResultHoldsNoWrapperOverARequestSurfaceItEchoes
{
	NSDictionary *properties = @{ (id)kIOSurfaceWidth: @16, (id)kIOSurfaceHeight: @16, (id)kIOSurfaceBytesPerElement: @4,
								  (id)kIOSurfacePixelFormat: @(kCVPixelFormatType_32BGRA) };
	IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)properties);
	XCTAssert(surface != NULL);
	if (surface == NULL) {
		return;
	}
	CFIndex baseline = CFGetRetainCount(surface);
	nfk_buffer buffer;
	memset(&buffer, 0, sizeof buffer);
	buffer.struct_size = (uint32_t)sizeof buffer;
	buffer.kind = NFK_BUFFER_SURFACE;
	buffer.surface = surface;
	buffer.length = 1024;
	buffer.descriptor = -1;
	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:NULL];
	nfk_payload request = [self payloadWithJSON:"{\"inputs\": {\"data\": {\"$nfk\": \"buffer\", \"index\": 0, \"as\": \"data\", \"length\": 1024}}, "
										"\"parameters\": {\"echoData\": true}}"
										buffers:&buffer count:1];
	nfk_error *error = NULL;
	nfk_result *result = nfk_backend_run(backend, &request, &error);
	XCTAssert(result != NULL, @"%s", nfk_error_message(error));
	NSDictionary *echo = [self outputsOfResult:result][@"echo"];
	XCTAssertEqualObjects(echo[@"$nfk"], @"buffer", @"the echoed data comes back as a copy in a buffer");
	XCTAssertEqual(CFGetRetainCount(surface), baseline, @"nothing the result keeps retains the surface or its read-only lock");
	nfk_result_free(result);
	nfk_backend_free(backend);
	CFRelease(surface);
}

- (void)testAFileBufferIsReadFromItsPath
{
	NSURL *file = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:
				   [NSString stringWithFormat:@"inferkit-c-api-%@.bin", NSUUID.UUID.UUIDString]];
	NSMutableData *contents = [NSMutableData dataWithLength:1000];
	memset(contents.mutableBytes, 2, contents.length);
	XCTAssertTrue([contents writeToURL:file atomically:YES]);

	nfk_buffer buffer;
	memset(&buffer, 0, sizeof buffer);
	buffer.struct_size = (uint32_t)sizeof buffer;
	buffer.kind = NFK_BUFFER_FILE;
	buffer.path = file.fileSystemRepresentation;
	buffer.descriptor = -1;
	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:NULL];
	nfk_payload request = [self payloadWithJSON:"{\"inputs\": {\"data\": {\"$nfk\": \"buffer\", \"index\": 0, \"as\": \"data\", \"length\": 1000}}}"
										buffers:&buffer count:1];
	nfk_error *error = NULL;
	nfk_result *result = nfk_backend_run(backend, &request, &error);
	XCTAssert(result != NULL, @"%s", nfk_error_message(error));
	XCTAssertEqualObjects([self outputsOfResult:result][@"dataSum"], @2000);
	nfk_result_free(result);
	nfk_backend_free(backend);
	[NSFileManager.defaultManager removeItemAtURL:file error:NULL];
}

- (void)testARequestInferKitCannotReadFailsWithoutRunning
{
	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:NULL];
	int ended = atomic_load(&NFKCAPIEchoRunsEnded);
	const char *unreadable[] = { "not json", "[1, 2]", "{\"parameters\": {}}" };
	for (size_t index = 0; index < sizeof unreadable / sizeof unreadable[0]; index++) {
		nfk_payload request = [self payloadWithJSON:unreadable[index] buffers:NULL count:0];
		nfk_error *error = NULL;
		XCTAssert(nfk_backend_run(backend, &request, &error) == NULL, @"%s", unreadable[index]);
		XCTAssertEqualObjects(@(nfk_error_domain(error)), NFKInferenceErrorDomain, @"%s", unreadable[index]);
		XCTAssertEqual(nfk_error_code(error), (int64_t)kNFKError_InferenceMissingInput, @"%s", unreadable[index]);
		nfk_error_free(error);
	}

	nfk_payload missingBuffer = [self payloadWithJSON:"{\"inputs\": {\"data\": {\"$nfk\": \"buffer\", \"index\": 3, \"as\": \"data\", \"length\": 4}}}"
											  buffers:NULL count:0];
	nfk_error *error = NULL;
	XCTAssert(nfk_backend_run(backend, &missingBuffer, &error) == NULL);
	XCTAssertEqualObjects(@(nfk_error_domain(error)), NFKInferenceErrorDomain);
	XCTAssertTrue([@(nfk_error_message(error)) containsString:@"buffer 3"], @"%s", nfk_error_message(error));
	nfk_error_free(error);

	nfk_payload shortStructure = [self payloadWithJSON:"{\"inputs\": {}}" buffers:NULL count:0];
	shortStructure.struct_size = 8;
	error = NULL;
	XCTAssert(nfk_backend_run(backend, &shortStructure, &error) == NULL);
	XCTAssertEqual(nfk_error_code(error), (int64_t)kNFKError_InferenceMissingInput);
	nfk_error_free(error);

	XCTAssertEqual(atomic_load(&NFKCAPIEchoRunsEnded), ended, @"no refused request reached the backend");
	nfk_backend_free(backend);
}

#pragma mark Value classes

- (void)testEveryValueClassComesBackAsJSON
{
	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:NULL];
	nfk_payload request = [self payloadWithJSON:"{\"inputs\": {}, \"parameters\": {\"valueObjects\": true}}" buffers:NULL count:0];
	nfk_error *error = NULL;
	nfk_result *result = nfk_backend_run(backend, &request, &error);
	XCTAssert(result != NULL, @"%s", nfk_error_message(error));
	const char *json = nfk_result_json(result);
	XCTAssert(json != NULL && strstr(json, "archive") == NULL, @"%s", json);
	NSDictionary *outputs = [self outputsOfResult:result];
	NSDictionary *detection = outputs[NFKOutputDetections][0];
	XCTAssertEqualObjects(detection[@"$nfk"], @"detection");
	XCTAssertEqualObjects(detection[@"label"], @"cat");
	XCTAssertEqualObjects(detection[@"boundingBox"], (@{ @"x": @0.25, @"y": @0.25, @"width": @0.5, @"height": @0.5 }));
	XCTAssertEqualObjects(detection[@"quadrilateral"][@"topLeft"], (@{ @"x": @0.25, @"y": @0.75 }));
	XCTAssertEqualObjects(outputs[NFKOutputPose][0][0][@"$nfk"], @"keypoint");
	XCTAssertEqualObjects(outputs[NFKOutputPose][0][0][@"position"], (@{ @"x": @0.5, @"y": @0.5 }));
	XCTAssertEqualObjects(outputs[NFKOutputClassifications][0][@"label"], @"dog");
	XCTAssertEqualObjects(outputs[NFKOutputSegments][0][@"$nfk"], @"audioSegment");
	XCTAssertEqualObjects(outputs[NFKOutputMIDI][@"$nfk"], @"midiSequence");
	XCTAssertEqualObjects(outputs[NFKOutputMIDI][@"notes"][0][@"pitch"], @60);
	XCTAssertEqualObjects(outputs[NFKOutputBeats][0][@"downbeat"], @YES);
	XCTAssertEqualObjects(outputs[NFKOutputAnswers][@"spam"][@"choice"], @"yes");
	XCTAssertEqual(nfk_result_buffer_count(result), 0u);
	nfk_result_free(result);
	nfk_backend_free(backend);
}

- (void)testAValueClassInARequestReachesTheBackend
{
	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:NULL];
	nfk_payload request = [self payloadWithJSON:"{\"inputs\": {\"region\": {\"$nfk\": \"detection\", \"label\": \"x\", "
										"\"boundingBox\": {\"x\": 0, \"y\": 0, \"width\": 1, \"height\": 1}}}}"
										buffers:NULL count:0];
	nfk_error *error = NULL;
	nfk_result *result = nfk_backend_run(backend, &request, &error);
	XCTAssert(result != NULL, @"%s", nfk_error_message(error));
	XCTAssertEqualObjects([self outputsOfResult:result][@"regionLabel"], @"x", @"the backend received an NFKDetection");
	nfk_result_free(result);

	nfk_payload malformed = [self payloadWithJSON:"{\"inputs\": {\"region\": {\"$nfk\": \"detection\", \"label\": \"x\"}}}"
										  buffers:NULL count:0];
	error = NULL;
	XCTAssert(nfk_backend_run(backend, &malformed, &error) == NULL);
	XCTAssertEqual(nfk_error_code(error), (int64_t)kNFKError_InferenceUnsupported);
	XCTAssertTrue([@(nfk_error_message(error)) containsString:@"boundingBox"], @"%s", nfk_error_message(error));
	nfk_error_free(error);
	nfk_backend_free(backend);
}

#pragma mark Preparing

- (void)testPreparingLoadsTheModel
{
	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:NULL];
	char *before = nfk_backend_copy_description(backend);
	XCTAssertEqualObjects([self JSONObject:before][@"loaded"], @NO);
	nfk_string_free(before);
	nfk_error *error = NULL;
	XCTAssertEqual(nfk_backend_prepare(backend, &error), 1);
	XCTAssert(error == NULL);
	char *description = nfk_backend_copy_description(backend);
	XCTAssertEqualObjects([self JSONObject:description][@"loaded"], @YES);
	nfk_string_free(description);
	nfk_backend_free(backend);
}

- (void)testAPreparationThatFailsReportsWhy
{
	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:"{\"failPrepare\": true}"];
	nfk_error *error = NULL;
	XCTAssertEqual(nfk_backend_prepare(backend, &error), 0);
	XCTAssertEqual(nfk_error_code(error), (int64_t)kNFKError_InferenceNotReady);
	XCTAssertEqualObjects(@(nfk_error_message(error)), @"the echo model is set to fail loading");
	nfk_error_free(error);
	nfk_backend_free(backend);

	error = NULL;
	XCTAssertEqual(nfk_backend_prepare(NULL, &error), 0);
	XCTAssert(error != NULL);
	nfk_error_free(error);
}

#pragma mark Error details

- (NSDictionary *)detailsOf:(const nfk_error *)error
{
	char *json = nfk_error_copy_details_json(error);
	NSDictionary *details = [self JSONObject:json];
	XCTAssertNotNil(details, @"%s", json);
	nfk_string_free(json);
	return details;
}

- (void)testInvalidOptionsListTheirProblems
{
	nfk_error *error = NULL;
	XCTAssert(nfk_backend_create(NFKCAPIEchoModel.UTF8String, "{\"nope\": 1}", &error) == NULL);
	NSArray *problems = [self detailsOf:error][NFKModelOptionProblemsKey];
	XCTAssertEqual(problems.count, 1u);
	XCTAssertEqualObjects(problems.firstObject[@"key"], @"nope");
	XCTAssertEqualObjects(problems.firstObject[@"code"], @"unknown-key");
	XCTAssertEqualObjects(problems.firstObject[@"value"], @1);
	XCTAssertGreaterThan([problems.firstObject[@"message"] length], 0u);
	nfk_error_free(error);
}

- (void)testARunErrorCarriesItsCauseAndLeavesOutWhatJSONCannotCarry
{
	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:NULL];
	nfk_payload request = [self payloadWithJSON:"{\"inputs\": {}, \"parameters\": {\"fail\": true}}" buffers:NULL count:0];
	nfk_error *error = NULL;
	XCTAssert(nfk_backend_run(backend, &request, &error) == NULL);
	NSDictionary *details = [self detailsOf:error];
	NSDictionary *expectedCause = @{ @"domain": @"com.example.cause", @"code": @7, @"message": @"the cause", @"details": @{} };
	XCTAssertEqualObjects(details[NSUnderlyingErrorKey], expectedCause);
	XCTAssertEqualObjects(details[@"note"], @"kept");
	XCTAssertNil(details[@"opaque"]);
	XCTAssertNil(details[NSLocalizedDescriptionKey], @"the message is not repeated");
	nfk_error_free(error);
	nfk_backend_free(backend);
}

- (void)testARefusalThatDoesNotFitCarriesItsEstimate
{
	nfk_error *error = NULL;
	XCTAssert(nfk_backend_create(NFKCAPIHugeModel.UTF8String, NULL, &error) == NULL);
	XCTAssertEqual(nfk_error_code(error), (int64_t)NFKModelCatalogErrorDoesNotFit);
	NSDictionary *estimate = [self detailsOf:error][NFKModelMemoryEstimateKey];
	XCTAssertTrue([estimate isKindOfClass:NSDictionary.class], @"%@", estimate);
	XCTAssertGreaterThan(estimate.count, 0u);
	nfk_error_free(error);
}

- (void)testAnErrorWithoutDetailsIsAnEmptyObject
{
	nfk_error *error = NULL;
	XCTAssert(nfk_backend_create(NULL, NULL, &error) == NULL);
	XCTAssertEqualObjects([self detailsOf:error], @{});
	nfk_error_free(error);
	XCTAssert(nfk_error_copy_details_json(NULL) == NULL);
}

#pragma mark Submitting

- (void)testASubmittedRunReportsProgressAndPartialsThenCompletesOnce
{
	nfk_backend *backend = [self backendFor:NFKCAPIStreamingModel options:NULL];
	NFKCAPICallbackLog *log = [[NFKCAPICallbackLog alloc] init];
	nfk_payload request = [self payloadWithJSON:"{\"inputs\": {}, \"parameters\": {\"steps\": 5, \"stepMilliseconds\": 10}}"
										buffers:NULL count:0];
	nfk_job *job = nfk_backend_submit(backend, &request, NFKCAPITestProgress, NFKCAPITestCompletion, (__bridge void *)log);
	XCTAssert(job != NULL);
	XCTAssertTrue([self wait:log.completed seconds:NFKCAPIDeadline]);
	nfk_job_free(job);

	XCTAssertEqual(log.completionCount, 1);
	XCTAssertGreaterThanOrEqual(log.progressCount, 1);
	NSDictionary *partial = [self JSONObject:log.lastPartialJSON.UTF8String][@"outputs"];
	XCTAssertEqualObjects(partial[@"step"], @5);
	XCTAssertEqualObjects(partial[NFKOutputDetections][0][@"$nfk"], @"detection", @"a partial carries value classes as JSON");
	XCTAssertEqualObjects([self JSONObject:log.resultJSON.UTF8String][@"outputs"][NFKOutputText], @"done");
	XCTAssertNil(log.errorDomain);
	XCTAssertFalse(log.overlapped, @"callbacks run one at a time");
	XCTAssertFalse(log.ranOnSubmittingThread);
	XCTAssertFalse(log.ranBelowUserInitiated);
	nfk_backend_free(backend);
}

- (void)testASynchronousBackendRunsThroughSubmit
{
	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:"{\"mode\": \"queued\"}"];
	NFKCAPICallbackLog *log = [[NFKCAPICallbackLog alloc] init];
	nfk_payload request = [self payloadWithJSON:"{\"inputs\": {\"prompt\": \"later\"}}" buffers:NULL count:0];
	nfk_job *job = nfk_backend_submit(backend, &request, NULL, NFKCAPITestCompletion, (__bridge void *)log);
	nfk_backend_free(backend);
	XCTAssertTrue([self wait:log.completed seconds:NFKCAPIDeadline], @"the job holds the backend it was submitted to");
	nfk_job_free(job);
	XCTAssertEqualObjects([self JSONObject:log.resultJSON.UTF8String][@"outputs"][NFKOutputText], @"queued:later");
	XCTAssertEqual(log.completionCount, 1);
}

- (void)testCancellingASubmittedRunCompletesWithACancellationError
{
	nfk_backend *backend = [self backendFor:NFKCAPIStreamingModel options:NULL];
	NFKCAPICallbackLog *log = [[NFKCAPICallbackLog alloc] init];
	nfk_payload request = [self payloadWithJSON:"{\"inputs\": {}, \"parameters\": {\"steps\": 1000, \"stepMilliseconds\": 20}}"
										buffers:NULL count:0];
	nfk_job *job = nfk_backend_submit(backend, &request, NFKCAPITestProgress, NFKCAPITestCompletion, (__bridge void *)log);
	XCTAssertTrue([self wait:log.progressed seconds:NFKCAPIDeadline]);
	nfk_job_cancel(job);
	XCTAssertTrue([self wait:log.completed seconds:NFKCAPIDeadline]);
	nfk_job_cancel(job);
	nfk_job_free(job);
	XCTAssertEqual(log.completionCount, 1);
	XCTAssertEqualObjects(log.errorDomain, NSCocoaErrorDomain);
	XCTAssertEqual(log.errorCode, (int64_t)NSUserCancelledError);
	XCTAssertNil(log.resultJSON);
	nfk_backend_free(backend);
}

- (void)testACancelledRunCompletesOnlyOnceItReleasesTheRequestBuffers
{
	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:NULL];
	NFKCAPICallbackLog *log = [[NFKCAPICallbackLog alloc] init];
	NSMutableData *data = [NSMutableData dataWithLength:256];
	nfk_buffer buffer = [self bytesBuffer:data.bytes length:data.length];
	nfk_payload request = [self payloadWithJSON:"{\"inputs\": {\"data\": {\"$nfk\": \"buffer\", \"index\": 0, \"as\": \"data\", \"length\": 256}}, "
										"\"parameters\": {\"sleepMilliseconds\": 300, \"signalStart\": true}}"
										buffers:&buffer count:1];
	int ended = atomic_load(&NFKCAPIEchoRunsEnded);
	nfk_job *job = nfk_backend_submit(backend, &request, NULL, NFKCAPITestCompletion, (__bridge void *)log);
	XCTAssertTrue([self wait:NFKCAPIEchoRunStarted seconds:NFKCAPIDeadline]);
	nfk_job_cancel(job);
	XCTAssertTrue([self wait:log.completed seconds:NFKCAPIDeadline]);
	nfk_job_free(job);
	XCTAssertEqual(log.errorCode, (int64_t)NSUserCancelledError);
	XCTAssertEqual(log.endedRunsAtCompletion, ended + 1, @"the completion waited for the run that reads the buffer");
	nfk_backend_free(backend);
}

/*! A malloc'd buffer of 4096 bytes of 1, whose sum is 4096, and the nfk_buffer over it. The caller frees it. */
- (uint8_t *)newSlowDataBuffer:(nfk_buffer *)outBuffer
{
	uint8_t *bytes = malloc(4096);
	memset(bytes, 1, 4096);
	*outBuffer = [self bytesBuffer:bytes length:4096];
	return bytes;
}

- (void)testFreeingAJobWaitsUntilTheRunNoLongerReadsItsBuffers
{
	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:NULL];
	NFKCAPICallbackLog *log = [[NFKCAPICallbackLog alloc] init];
	nfk_buffer buffer;
	uint8_t *bytes = [self newSlowDataBuffer:&buffer];
	nfk_payload request = [self payloadWithJSON:NFKCAPISlowDataRequest buffers:&buffer count:1];
	int ended = atomic_load(&NFKCAPIEchoRunsEnded);
	atomic_store(&NFKCAPIEchoLastDataSum, 0);
	nfk_job *job = nfk_backend_submit(backend, &request, NULL, NFKCAPITestCompletion, (__bridge void *)log);
	XCTAssertTrue([self wait:NFKCAPIEchoRunStarted seconds:NFKCAPIDeadline]);
	nfk_job_free(job);
	XCTAssertEqual(atomic_load(&NFKCAPIEchoRunsEnded), ended + 1, @"the backend read the buffer before nfk_job_free returned");
	XCTAssertEqual(atomic_load(&NFKCAPIEchoLastDataSum), 4096ull, @"it read the caller's bytes, not freed memory");
	memset(bytes, 0xFF, 4096);
	free(bytes);
	XCTAssertFalse([self wait:log.completed seconds:0.3], @"a freed job's completion never fires");
	nfk_backend_free(backend);
}

- (void)testFreeingACancelledJobBeforeItsCompletionWaitsForTheRun
{
	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:NULL];
	NFKCAPICallbackLog *log = [[NFKCAPICallbackLog alloc] init];
	nfk_buffer buffer;
	uint8_t *bytes = [self newSlowDataBuffer:&buffer];
	nfk_payload request = [self payloadWithJSON:NFKCAPISlowDataRequest buffers:&buffer count:1];
	int ended = atomic_load(&NFKCAPIEchoRunsEnded);
	atomic_store(&NFKCAPIEchoLastDataSum, 0);
	nfk_job *job = nfk_backend_submit(backend, &request, NULL, NFKCAPITestCompletion, (__bridge void *)log);
	XCTAssertTrue([self wait:NFKCAPIEchoRunStarted seconds:NFKCAPIDeadline]);
	nfk_job_cancel(job);
	nfk_job_free(job);
	XCTAssertEqual(atomic_load(&NFKCAPIEchoRunsEnded), ended + 1, @"the backend, which reads after the cancel, finished first");
	XCTAssertEqual(atomic_load(&NFKCAPIEchoLastDataSum), 4096ull);
	memset(bytes, 0xFF, 4096);
	free(bytes);
	XCTAssertEqual(log.completionCount, 0);
	nfk_backend_free(backend);
}

- (void)testFreeingAJobInsideItsProgressCallbackWaitsForTheRun
{
	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:NULL];
	NFKCAPICallbackLog *log = [[NFKCAPICallbackLog alloc] init];
	nfk_buffer buffer;
	uint8_t *bytes = [self newSlowDataBuffer:&buffer];
	nfk_payload request = [self payloadWithJSON:NFKCAPISlowDataRequest buffers:&buffer count:1];
	int ended = atomic_load(&NFKCAPIEchoRunsEnded);
	[log.lock lock];
	nfk_job *job = nfk_backend_submit(backend, &request, NFKCAPITestProgress, NFKCAPITestCompletion, (__bridge void *)log);
	log.jobToFreeInProgress = job;
	[log.lock unlock];
	XCTAssertTrue([self wait:log.progressed seconds:NFKCAPIDeadline], @"the run reports its start, and the callback frees the job");
	XCTAssertTrue([self wait:NFKCAPIEchoRunStarted seconds:NFKCAPIDeadline]);
	XCTAssertEqual(log.endedRunsAtFree, ended + 1, @"nfk_job_free inside the callback returned after the run");
	memset(bytes, 0xFF, 4096);
	free(bytes);
	XCTAssertFalse([self wait:log.completed seconds:0.3], @"the callback that freed the job was its last");
	XCTAssertEqual(log.progressCount, 1);
	nfk_backend_free(backend);
}

- (void)testFreeingAJobBeforeItEndsStopsEveryCallback
{
	nfk_backend *backend = [self backendFor:NFKCAPIStreamingModel options:NULL];
	NFKCAPICallbackLog *log = [[NFKCAPICallbackLog alloc] init];
	nfk_payload request = [self payloadWithJSON:"{\"inputs\": {}, \"parameters\": {\"steps\": 1000, \"stepMilliseconds\": 5}}"
										buffers:NULL count:0];
	nfk_job *job = nfk_backend_submit(backend, &request, NFKCAPITestProgress, NFKCAPITestCompletion, (__bridge void *)log);
	XCTAssertTrue([self wait:log.progressed seconds:NFKCAPIDeadline]);
	nfk_job_free(job);
	[log markFreed];
	XCTAssertFalse([self wait:log.completed seconds:0.5], @"a freed job's completion never fires");
	XCTAssertEqual(log.callbacksAfterFree, 0);
	XCTAssertEqual(log.completionCount, 0);
	nfk_backend_free(backend);
}

- (void)testAJobFreedInsideItsCompletionReturnsAtOnce
{
	nfk_backend *backend = [self backendFor:NFKCAPIStreamingModel options:NULL];
	NFKCAPICallbackLog *log = [[NFKCAPICallbackLog alloc] init];
	nfk_payload request = [self payloadWithJSON:"{\"inputs\": {}, \"parameters\": {\"steps\": 3, \"stepMilliseconds\": 50}}"
										buffers:NULL count:0];
	[log.lock lock];
	nfk_job *job = nfk_backend_submit(backend, &request, NFKCAPITestProgress, NFKCAPITestCompletion, (__bridge void *)log);
	log.jobToFreeInCompletion = job;
	[log.lock unlock];
	XCTAssertTrue([self wait:log.completed seconds:NFKCAPIDeadline], @"freeing the job from its own completion does not wait on itself");
	XCTAssertEqual(log.completionCount, 1);
	nfk_backend_free(backend);
}

- (void)testAnUnreadableSubmittedRequestCompletesWithItsError
{
	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:NULL];
	NFKCAPICallbackLog *log = [[NFKCAPICallbackLog alloc] init];
	nfk_payload request = [self payloadWithJSON:"{]" buffers:NULL count:0];
	nfk_job *job = nfk_backend_submit(backend, &request, NFKCAPITestProgress, NFKCAPITestCompletion, (__bridge void *)log);
	XCTAssert(job != NULL);
	XCTAssertTrue([self wait:log.completed seconds:NFKCAPIDeadline]);
	nfk_job_free(job);
	XCTAssertEqualObjects(log.errorDomain, NFKInferenceErrorDomain);
	XCTAssertEqual(log.errorCode, (int64_t)kNFKError_InferenceMissingInput);
	XCTAssertFalse(log.ranOnSubmittingThread);
	nfk_backend_free(backend);
}

#pragma mark Ownership

- (void)testEveryFreeAcceptsNullAndEveryReaderAnswersNull
{
	nfk_backend_free(NULL);
	nfk_backend_unload(NULL);
	nfk_job_cancel(NULL);
	nfk_job_free(NULL);
	nfk_result_free(NULL);
	nfk_error_free(NULL);
	nfk_string_free(NULL);
	XCTAssert(nfk_backend_copy_description(NULL) == NULL);
	XCTAssert(nfk_result_json(NULL) == NULL);
	XCTAssertEqual(nfk_result_buffer_count(NULL), 0u);
	XCTAssert(nfk_result_buffer(NULL, 0) == NULL);
	XCTAssert(nfk_error_domain(NULL) == NULL);
	XCTAssert(nfk_error_message(NULL) == NULL);
	XCTAssertEqual(nfk_error_code(NULL), (int64_t)0);

	nfk_payload request = [self payloadWithJSON:"{\"inputs\": {}}" buffers:NULL count:0];
	nfk_error *error = NULL;
	XCTAssert(nfk_backend_run(NULL, &request, &error) == NULL);
	XCTAssert(error != NULL, @"a run without a backend reports why");
	nfk_error_free(error);
	XCTAssert(nfk_backend_submit(NULL, &request, NULL, NFKCAPITestCompletion, NULL) == NULL);

	nfk_backend *backend = [self backendFor:NFKCAPIEchoModel options:NULL];
	XCTAssert(nfk_backend_submit(backend, NULL, NULL, NFKCAPITestCompletion, NULL) == NULL);
	XCTAssert(nfk_backend_submit(backend, &request, NULL, NULL, NULL) == NULL);
	nfk_backend_unload(backend);
	nfk_backend_free(backend);
}

- (void)testAHandleOfAnotherTypeReadsAsNull
{
	nfk_error *error = NULL;
	XCTAssert(nfk_backend_create("c-api.ghost", NULL, &error) == NULL);
	XCTAssert(nfk_result_json((const nfk_result *)(const void *)error) == NULL);
	XCTAssert(nfk_backend_copy_description((nfk_backend *)(void *)error) == NULL);
	nfk_result_free((nfk_result *)(void *)error);
	XCTAssert(nfk_error_message(error) != NULL, @"freeing it as another type left the error alone");
	nfk_error_free(error);
}

#pragma mark Plugins

- (void)testANullPluginDirectoryFails
{
	nfk_error *error = NULL;
	XCTAssertEqual(nfk_plugins_add_directory(NULL, &error), -1);
	XCTAssert(error != NULL);
	nfk_error_free(error);
}

#if TARGET_OS_OSX

- (void)testAnEmptyPluginDirectoryAddsNothing
{
	NSURL *directory = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:
						[@"inferkit-c-api-plugins-" stringByAppendingString:NSUUID.UUID.UUIDString] isDirectory:YES];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:NULL]);
	nfk_error *error = NULL;
	XCTAssertEqual(nfk_plugins_add_directory(directory.fileSystemRepresentation, &error), 0);
	XCTAssert(error == NULL, @"%s", nfk_error_message(error));
	nfk_error_free(error);
	[NSFileManager.defaultManager removeItemAtURL:directory error:NULL];
}

#pragma mark Programs in C and C++

/*! Builds source as a bundle whose InferKit symbols bind to this process, loads it, and calls its
	run function with the echo model. */
- (void)assertConsumer:(NSString *)source compiler:(NSString *)compiler standard:(NSString *)standard symbol:(NSString *)symbol
{
	NSURL *root = NFKPluginFixtures.sourceRootURL;
	NSURL *directory = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:
						[@"inferkit-c-consumer-" stringByAppendingString:NSUUID.UUID.UUIDString] isDirectory:YES];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:NULL]);
	NSString *bundle = [directory URLByAppendingPathComponent:@"consumer.bundle"].path;
#if defined(__arm64__)
	NSString *architecture = @"arm64";
#else
	NSString *architecture = @"x86_64";
#endif
	NSArray<NSString *> *arguments = @[ compiler, standard, @"-pedantic", @"-Wall", @"-Wextra", @"-Werror", @"-bundle",
										@"-undefined", @"dynamic_lookup", @"-arch", architecture, @"-mmacosx-version-min=11.0",
										@"-I", [root URLByAppendingPathComponent:@"Sources/InferKitC/include"].path,
										[root URLByAppendingPathComponent:source].path, @"-o", bundle ];
	NSError *error = nil;
	NSString *output = [NFKPluginFixtures runProgram:@"/usr/bin/xcrun" arguments:arguments error:&error];
	XCTAssertNotNil(output, @"%@", error.localizedDescription);
	XCTAssertFalse([output containsString:@"warning:"], @"%@", output);
	if (output == nil) {
		return;
	}
	void *image = dlopen(bundle.fileSystemRepresentation, RTLD_NOW | RTLD_LOCAL);
	XCTAssert(image != NULL, @"%s", dlerror());
	int (*run)(const char *, char *, size_t) = image != NULL ? (int (*)(const char *, char *, size_t))dlsym(image, symbol.UTF8String) : NULL;
	XCTAssert(run != NULL, @"%@ is not exported", symbol);
	if (run == NULL) {
		return;
	}
	char report[512] = { 0 };
	XCTAssertEqual(run(NFKCAPIEchoModel.UTF8String, report, sizeof report), 0, @"%s", report);
	XCTAssertEqualObjects(@(report), @"ok");
}

- (void)testAC99ProgramBuildsAndRunsAModel
{
	[self assertConsumer:@"Tests/PluginFixtures/headers/inferkit_c_consumer.c" compiler:@"clang" standard:@"-std=c99"
				  symbol:@"inferkit_c_consumer_run"];
}

- (void)testACPlusPlus11ProgramBuildsAndRunsAModel
{
	[self assertConsumer:@"Tests/PluginFixtures/headers/inferkit_cpp_consumer.cpp" compiler:@"clang++" standard:@"-std=c++11"
				  symbol:@"inferkit_cpp_consumer_run"];
}

#endif

@end
