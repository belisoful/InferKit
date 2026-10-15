//
//  NFKBackendLifecycleTests.m
//  InferKitTests
//
//  The protocol's unload and memory figures: the uniform helpers over a backend that implements them and
//  one that does not, and the figures a server reports for a hosted model, from the backend or from a
//  registered runtime's allocation readings around the build.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

/*! The server's clock, which a test replaces. */
@interface NFKInferenceServer (NFKLifecycleTesting)
@property (nonatomic, copy) NSTimeInterval (^uptimeProvider)(void);
@end

/*! A backend that unloads, counting its unloads and loads, and reports fixed memory figures while loaded. */
@interface NFKLifecycleTestBackend : NSObject <NFKInferenceBackend>
@property (atomic, assign) BOOL loaded;
@property (atomic, assign) NSInteger unloads;
@property (atomic, assign) NSInteger loads;
@property (nonatomic, assign) NSInteger measuredBytes;
@property (nonatomic, assign) NSInteger measuredPeak;
@end

@implementation NFKLifecycleTestBackend

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_loaded = YES;
	}
	return self;
}

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"lifecycle-test";
}

- (BOOL)prepareWithError:(NSError **)error
{
	if (!self.loaded) {
		self.loaded = YES;
		self.loads += 1;
	}
	return YES;
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	[self prepareWithError:error];
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"ok" }];
}

- (void)unload
{
	self.loaded = NO;
	self.unloads += 1;
}

- (BOOL)isLoaded
{
	return self.loaded;
}

- (NSInteger)residentByteCount
{
	return self.loaded ? self.measuredBytes : 0;
}

- (NSInteger)peakByteCount
{
	return self.measuredPeak;
}

@end

/*! A backend with none of the lifecycle members. */
@interface NFKLifecyclePlainBackend : NSObject <NFKInferenceBackend>
@property (nonatomic, assign) BOOL ready;
@end

@implementation NFKLifecyclePlainBackend

- (BOOL)isReady
{
	return self.ready;
}

- (NSString *)backendIdentifier
{
	return @"lifecycle-plain";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"ok" }];
}

@end

static NSInteger NFKLifecycleRuntimeAllocated = -1;
static NSInteger NFKLifecycleRuntimePeak = -1;

/*! A runtime whose allocation readings a test sets; it reads -1, which a server skips, outside these tests. */
@interface NFKLifecycleTestRuntime : NSObject <NFKServingRuntimeStatus>
@end

@implementation NFKLifecycleTestRuntime

+ (NSString *)runtimeStatusName
{
	return @"lifecycle-test-runtime";
}

+ (NSDictionary<NSString *, id> *)runtimeStatus
{
	return @{};
}

+ (NSInteger)allocatedByteCount
{
	@synchronized (self) {
		return NFKLifecycleRuntimeAllocated;
	}
}

+ (NSInteger)peakAllocatedByteCount
{
	@synchronized (self) {
		return NFKLifecycleRuntimePeak;
	}
}

+ (void)resetPeakAllocatedByteCount
{
	@synchronized (self) {
		NFKLifecycleRuntimePeak = NFKLifecycleRuntimeAllocated;
	}
}

+ (void)setAllocated:(NSInteger)allocated peak:(NSInteger)peak
{
	@synchronized (self) {
		NFKLifecycleRuntimeAllocated = allocated;
		NFKLifecycleRuntimePeak = peak;
	}
}

@end

@interface NFKBackendLifecycleTests : XCTestCase
@property (atomic, assign) NSTimeInterval now;
@end

@implementation NFKBackendLifecycleTests

- (void)tearDown
{
	[NFKLifecycleTestRuntime setAllocated:-1 peak:-1];
	[super tearDown];
}

/*! A server on the test's clock with its periodic check off, hosting provider under "model", loaded. */
- (NFKInferenceServer *)serverLoadingProvider:(NFKServedBackendProvider)provider
{
	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.residencyCheckInterval = 0;
	self.now = 1000;
	__weak NFKBackendLifecycleTests *weakSelf = self;
	server.uptimeProvider = ^NSTimeInterval {
		return weakSelf.now;
	};
	[server addBackendProvider:provider forModelName:@"model"];
	XCTestExpectation *loaded = [self expectationWithDescription:@"loaded"];
	[server loadModelNamed:@"model" completionHandler:^(NSError *error) {
		XCTAssertNil(error);
		[loaded fulfill];
	}];
	[self waitForExpectations:@[ loaded ] timeout:10];
	return server;
}

- (nullable NFKServerModelStatus *)statusOfModelIn:(NFKInferenceServer *)server
{
	for (NFKServerModelStatus *model in [server currentStatus].models) {
		if ([model.name isEqualToString:@"model"]) {
			return model;
		}
	}
	return nil;
}

#pragma mark Protocol helpers

- (void)testTheHelpersReachABackendThatUnloads
{
	NFKLifecycleTestBackend *backend = [[NFKLifecycleTestBackend alloc] init];
	backend.measuredBytes = 4096;
	backend.measuredPeak = 8192;
	XCTAssertTrue(NFKInferenceIsLoaded(backend));
	XCTAssertEqual(NFKInferenceResidentByteCount(backend), 4096);
	XCTAssertEqual(NFKInferencePeakByteCount(backend), 8192);

	NFKInferenceUnload(backend);
	XCTAssertEqual(backend.unloads, 1);
	XCTAssertFalse(NFKInferenceIsLoaded(backend));
	XCTAssertEqual(NFKInferenceResidentByteCount(backend), 0);
	XCTAssertEqual(NFKInferencePeakByteCount(backend), 8192, @"the peak describes the last load");

	NSError *error = nil;
	XCTAssertTrue(NFKInferencePrepare(backend, &error), @"%@", error);
	XCTAssertTrue(NFKInferenceIsLoaded(backend));
	XCTAssertEqual(backend.loads, 1);
}

- (void)testTheHelpersDefaultForABackendWithoutTheMembers
{
	NFKLifecyclePlainBackend *backend = [[NFKLifecyclePlainBackend alloc] init];
	backend.ready = YES;
	NFKInferenceUnload(backend);
	XCTAssertTrue(NFKInferenceIsLoaded(backend), @"isLoaded falls back to isReady");
	XCTAssertEqual(NFKInferenceResidentByteCount(backend), 0);
	XCTAssertEqual(NFKInferencePeakByteCount(backend), 0);
	backend.ready = NO;
	XCTAssertFalse(NFKInferenceIsLoaded(backend));
}

#pragma mark Server

- (void)testAServedModelReportsItsBackendsOwnFigures
{
	NFKLifecycleTestBackend *backend = [[NFKLifecycleTestBackend alloc] init];
	backend.measuredBytes = 4096;
	backend.measuredPeak = 8192;
	NFKInferenceServer *server = [self serverLoadingProvider:^id<NFKInferenceBackend>(NSError **error) {
		return backend;
	}];
	NFKServerModelStatus *status = [self statusOfModelIn:server];
	XCTAssertEqualObjects(status.residentBytes, @4096);
	XCTAssertEqualObjects(status.peakBytes, @8192);
}

- (void)testTheServerMeasuresABuildThroughARegisteredRuntime
{
	[NFKInferenceServer registerRuntimeStatusClassName:@"NFKLifecycleTestRuntime"];
	[NFKLifecycleTestRuntime setAllocated:1000 peak:5000];
	NFKInferenceServer *server = [self serverLoadingProvider:^id<NFKInferenceBackend>(NSError **error) {
		[NFKLifecycleTestRuntime setAllocated:1600 peak:2500];
		NFKLifecyclePlainBackend *backend = [[NFKLifecyclePlainBackend alloc] init];
		backend.ready = YES;
		return backend;
	}];
	NFKServerModelStatus *status = [self statusOfModelIn:server];
	XCTAssertEqualObjects(status.residentBytes, @600, @"the rise in allocated bytes over the build");
	XCTAssertEqualObjects(status.peakBytes, @1500, @"the peak since the reset at the build's start");
}

- (void)testAModelNothingMeasuredReportsNoFigures
{
	NFKInferenceServer *server = [self serverLoadingProvider:^id<NFKInferenceBackend>(NSError **error) {
		NFKLifecyclePlainBackend *backend = [[NFKLifecyclePlainBackend alloc] init];
		backend.ready = YES;
		return backend;
	}];
	NFKServerModelStatus *status = [self statusOfModelIn:server];
	XCTAssertNil(status.residentBytes);
	XCTAssertNil(status.peakBytes);
}

- (void)testUnloadingAServedModelUnloadsItsBackendAndKeepsTheFigures
{
	NFKLifecycleTestBackend *backend = [[NFKLifecycleTestBackend alloc] init];
	backend.measuredBytes = 4096;
	backend.measuredPeak = 4096;
	NFKInferenceServer *server = [self serverLoadingProvider:^id<NFKInferenceBackend>(NSError **error) {
		return backend;
	}];
	XCTAssertTrue([server unloadModelNamed:@"model"]);
	XCTAssertEqual(backend.unloads, 1, @"the server unloads the backend before it releases it");
	NFKServerModelStatus *status = [self statusOfModelIn:server];
	XCTAssertEqual(status.residency, NFKServerModelResidencyNotLoaded);
	XCTAssertEqualObjects(status.residentBytes, @4096, @"the figures describe the last load");
}

- (void)testAnIdleUnloadUnloadsTheBackend
{
	NFKLifecycleTestBackend *backend = [[NFKLifecycleTestBackend alloc] init];
	NFKInferenceServer *server = [self serverLoadingProvider:^id<NFKInferenceBackend>(NSError **error) {
		return backend;
	}];
	server.unloadGracePeriod = 0;
	server.idleUnloadInterval = 60;
	self.now += 61;
	XCTAssertEqualObjects([server unloadIdleModels], @[ @"model" ]);
	XCTAssertEqual(backend.unloads, 1);
}

- (void)testRemovingOrReplacingAModelUnloadsItsBackend
{
	NFKLifecycleTestBackend *removed = [[NFKLifecycleTestBackend alloc] init];
	NFKInferenceServer *server = [self serverLoadingProvider:^id<NFKInferenceBackend>(NSError **error) {
		return removed;
	}];
	[server removeBackendForModelName:@"model"];
	XCTAssertEqual(removed.unloads, 1, @"a removed model's backend is unloaded");

	NFKLifecycleTestBackend *replaced = [[NFKLifecycleTestBackend alloc] init];
	NFKLifecycleTestBackend *replacement = [[NFKLifecycleTestBackend alloc] init];
	[server addBackend:replaced forModelName:@"hosted"];
	[server addBackend:replaced forModelName:@"hosted"];
	XCTAssertEqual(replaced.unloads, 0, @"re-hosting the same backend keeps it loaded");
	[server addBackend:replacement forModelName:@"hosted"];
	XCTAssertEqual(replaced.unloads, 1, @"a replaced model's backend is unloaded");
	XCTAssertEqual(replacement.unloads, 0);
}

- (void)testTheStatusReadsTheResidencyFigures
{
	NSDictionary *reply = @{ @"object": @"inferkit.status",
							 @"models": @[ @{ @"id": @"chat",
											  @"residency": @{ @"state": @"not_loaded", @"resident_bytes": @123,
															   @"peak_bytes": @456 } },
										   @{ @"id": @"bare" } ] };
	NFKServerStatus *status = [NFKServerStatus statusWithJSONObject:reply];
	XCTAssertEqualObjects(status.models.firstObject.residentBytes, @123);
	XCTAssertEqualObjects(status.models.firstObject.peakBytes, @456);
	XCTAssertNil(status.models.lastObject.residentBytes, @"a server that measured nothing");
}

@end
