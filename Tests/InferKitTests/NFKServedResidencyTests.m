//
//  NFKServedResidencyTests.m
//  InferKitTests
//
//  Models a server builds on demand and unloads when idle or under memory pressure. The server's
//  clock and pressure reading are the test's, and its periodic check is off, so each test drives the
//  unloading options itself.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

/*! The server's clock, which a test replaces. */
@interface NFKInferenceServer (NFKResidencyTesting)
@property (nonatomic, copy) NSTimeInterval (^uptimeProvider)(void);
@end

/*! A synchronous backend that answers "ok", optionally holding each run until the test opens its gate. */
@interface NFKResidencyTestBackend : NSObject <NFKInferenceBackend>
@property (nonatomic, strong, nullable) dispatch_semaphore_t gate;
@property (atomic, assign) NSInteger running;
@end

@implementation NFKResidencyTestBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"residency-test";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	self.running += 1;
	if (self.gate != nil) {
		dispatch_semaphore_wait(self.gate, DISPATCH_TIME_FOREVER);
	}
	self.running -= 1;
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"ok" }];
}

@end

static NSInteger NFKReliefRuntimeCached = 0;

/*! A runtime that, as MLX does, keeps what a backend releases in its cache until it returns its cache. A
	test registers it for the rest of the process, since registration has no undo; it holds nothing outside
	the test that sets its cache. */
@interface NFKReliefTestRuntime : NSObject <NFKServingRuntimeStatus>
+ (void)setCachedByteCount:(NSInteger)bytes;
@end

@implementation NFKReliefTestRuntime

+ (NSString *)runtimeStatusName
{
	return @"relief-test-runtime";
}

+ (NSDictionary<NSString *, id> *)runtimeStatus
{
	return @{};
}

+ (NSInteger)cachedByteCount
{
	@synchronized (self) {
		return NFKReliefRuntimeCached;
	}
}

+ (void)setCachedByteCount:(NSInteger)bytes
{
	@synchronized (self) {
		NFKReliefRuntimeCached = bytes;
	}
}

+ (void)releaseCachedMemory
{
	[self setCachedByteCount:0];
}

@end

/*! A backend that reports its resident bytes and releases conversation and expert caches it holds. */
@interface NFKReliefTestBackend : NFKResidencyTestBackend
@property (atomic, assign) NSInteger residentBytesHeld;
/*! Whether the conversation caches it releases go into NFKReliefTestRuntime's cache. */
@property (atomic, assign) BOOL parksReleasedCaches;
@property (atomic, assign) NSInteger conversationBytes;
@property (atomic, assign) NSInteger expertBytes;
@end

@implementation NFKReliefTestBackend

- (NSInteger)residentByteCount
{
	return self.residentBytesHeld;
}

- (NSInteger)releaseIdleConversationCaches
{
	NSInteger held = self.conversationBytes;
	self.conversationBytes = 0;
	if (self.parksReleasedCaches) {
		[NFKReliefTestRuntime setCachedByteCount:[NFKReliefTestRuntime cachedByteCount] + held];
	}
	return held;
}

- (NSInteger)releaseExpertCacheByteCount:(NSInteger)byteCount keepingByteCount:(NSInteger)keptByteCount
{
	NSInteger freed = MAX(MIN(byteCount, self.expertBytes - keptByteCount), 0);
	self.expertBytes -= freed;
	return freed;
}

@end

static NSString * const NFKAdmissionTestEngine = @"admission-test-engine";

/*! The engine's builder, which records the residency a build asks for and the budget it builds under, and
	builds a backend holding what that residency holds of a model with two 3,500-byte stages. */
@interface NFKAdmissionTestBuilder : NSObject <NFKModelBackendBuilding>
@property (atomic, copy, nullable) NSString *residency;
@property (atomic, strong, nullable) NFKModelMemoryBudget *budget;
@end

@implementation NFKAdmissionTestBuilder

- (nullable id<NFKInferenceBackend>)backendForBuildRequest:(NFKModelBuildRequest *)request error:(NSError **)error
{
	self.residency = request.loadOptions[NFKModelLoadOptionResidency];
	self.budget = [NFKModelMemoryBudget currentBudget];
	NFKReliefTestBackend *backend = [[NFKReliefTestBackend alloc] init];
	backend.residentBytesHeld = [self.residency isEqualToString:NFKModelResidencyStaged] ? 3500 : 7000;
	return backend;
}

@end

@interface NFKServedResidencyTests : XCTestCase
@property (atomic, assign) NSTimeInterval now;
@property (atomic, assign) double pressure;
@property (atomic, assign) NSInteger builds;
/*! Whether testABuildUnderWayCountsWhatAdmissionReservedForIt's provider waits at its gate. */
@property (atomic, assign) BOOL gatesBuilds;
@property (nonatomic, assign) NSUInteger savedRetryAttempts;
@end

@implementation NFKServedResidencyTests

- (void)setUp
{
	[super setUp];
	self.savedRetryAttempts = NFKRemoteTransport.retryAttempts;
	NFKRemoteTransport.retryAttempts = 0;
	self.now = 1000;
	self.pressure = 0;
	self.builds = 0;
}

- (void)tearDown
{
	NFKRemoteTransport.retryAttempts = self.savedRetryAttempts;
	[super tearDown];
}

#pragma mark Helpers

/*! A loopback server on the test's clock and pressure, with no periodic check and no grace period. */
- (NFKInferenceServer *)server
{
	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.port = 0;
	server.loopbackOnly = YES;
	server.residencyCheckInterval = 0;
	server.unloadGracePeriod = 0;
	__weak NFKServedResidencyTests *weakSelf = self;
	server.uptimeProvider = ^NSTimeInterval {
		return weakSelf.now;
	};
	server.memoryPressureProvider = ^double {
		return weakSelf.pressure;
	};
	return server;
}

- (NFKInferenceServer *)startedServer:(NFKInferenceServer *)server
{
	NSError *error = nil;
	XCTAssertTrue([server startWithError:&error], @"%@", error);
	[self addTeardownBlock:^{
		[server stop];
	}];
	return server;
}

/*! A provider that counts its builds and returns backend. */
- (NFKServedBackendProvider)providerReturning:(id<NFKInferenceBackend>)backend
{
	__weak NFKServedResidencyTests *weakSelf = self;
	return ^id<NFKInferenceBackend>(NSError **error) {
		weakSelf.builds += 1;
		return backend;
	};
}

- (void)load:(NSString *)modelName on:(NFKInferenceServer *)server
{
	XCTestExpectation *loaded = [self expectationWithDescription:modelName];
	[server loadModelNamed:modelName completionHandler:^(NSError *error) {
		XCTAssertNil(error);
		[loaded fulfill];
	}];
	[self waitForExpectations:@[ loaded ] timeout:10];
}

- (nullable NFKInferenceResult *)run:(NSString *)modelName on:(NFKInferenceServer *)server error:(NSError **)error
{
	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:server.localBaseURL];
	client.modelName = modelName;
	return [client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }] error:error];
}

- (nullable id)JSONAt:(NSString *)path on:(NFKInferenceServer *)server
{
	NSURL *url = [server.localBaseURL URLByAppendingPathComponent:path];
	NSData *data = [NFKRemoteTransport sendRequest:[NSURLRequest requestWithURL:url] session:NSURLSession.sharedSession
										  response:NULL error:NULL];
	return data != nil ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
}

// XCTNSPredicateExpectation evaluates about once a second, longer than the sub-second intervals these
// tests set, so the wait polls every 10 ms.
- (void)waitUntil:(BOOL (^)(void))condition description:(NSString *)description
{
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:10];
	while (!condition() && deadline.timeIntervalSinceNow > 0) {
		[NSRunLoop.currentRunLoop runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.01]];
	}
	XCTAssertTrue(condition(), @"%@", description);
}

#pragma mark Lazy hosting

- (void)testAProvidedModelIsListedWithoutBeingBuilt
{
	NFKInferenceServer *server = [self startedServer:[self server]];
	[server addBackendProvider:[self providerReturning:[[NFKResidencyTestBackend alloc] init]] forModelName:@"lazy"];

	NSDictionary *list = [self JSONAt:@"models" on:server];
	NSDictionary *entry = [list[@"data"] firstObject];
	XCTAssertEqualObjects(entry[@"id"], @"lazy");
	XCTAssertEqualObjects(entry[@"inferkit"][@"residency"][@"state"], @"not_loaded");
	XCTAssertEqualObjects(entry[@"inferkit"][@"ready"], @YES, @"a request builds the model, so it can be served");
	NSDictionary *single = [self JSONAt:@"models/lazy" on:server];
	XCTAssertEqualObjects(single[@"inferkit"][@"residency"][@"state"], @"not_loaded");

	NSError *error = nil;
	NFKServerModelStatus *status = [[NFKServerStatus fetchFromBaseURL:server.localBaseURL apiKey:nil error:&error] modelNamed:@"lazy"];
	XCTAssertNotNil(status, @"%@", error);
	XCTAssertEqual(status.residency, NFKServerModelResidencyNotLoaded);
	XCTAssertTrue(status.isUnloadable);
	XCTAssertEqual(status.loads, 0);
	XCTAssertNil(status.lastUsedDate);

	XCTAssertEqual(self.builds, 0, @"listing builds nothing");
	XCTAssertEqual([server residencyForModelName:@"lazy"], NFKServerModelResidencyNotLoaded);
	XCTAssertNil([server backendForModelName:@"lazy"]);
	XCTAssertEqual([server residencyForModelName:@"absent"], NFKServerModelResidencyUnknown);
}

- (void)testTheFirstRequestBuildsTheModel
{
	NFKInferenceServer *server = [self startedServer:[self server]];
	[server addBackendProvider:[self providerReturning:[[NFKResidencyTestBackend alloc] init]] forModelName:@"lazy"];

	NSError *error = nil;
	XCTAssertEqualObjects([self run:@"lazy" on:server error:&error].text, @"ok", @"%@", error);
	XCTAssertEqualObjects([self run:@"lazy" on:server error:&error].text, @"ok", @"%@", error);
	XCTAssertEqual(self.builds, 1, @"the second request finds the model resident");
	XCTAssertEqual([server residencyForModelName:@"lazy"], NFKServerModelResidencyResident);
	XCTAssertNotNil([server backendForModelName:@"lazy"]);

	NFKServerModelStatus *status = [[NFKServerStatus fetchFromBaseURL:server.localBaseURL apiKey:nil error:&error] modelNamed:@"lazy"];
	XCTAssertEqual(status.residency, NFKServerModelResidencyResident);
	XCTAssertEqualObjects(status.backendIdentifier, @"residency-test");
	XCTAssertEqual(status.loads, 1);
	XCTAssertNotNil(status.loadSeconds);
	XCTAssertNotNil(status.lastUsedDate);
	XCTAssertEqual(status.completed, 2);
}

- (void)testConcurrentRequestsShareOneBuild
{
	NFKInferenceServer *server = [self startedServer:[self server]];
	dispatch_semaphore_t entered = dispatch_semaphore_create(0);
	dispatch_semaphore_t proceed = dispatch_semaphore_create(0);
	NFKResidencyTestBackend *backend = [[NFKResidencyTestBackend alloc] init];
	__weak NFKServedResidencyTests *weakSelf = self;
	[server addBackendProvider:^id<NFKInferenceBackend>(NSError **error) {
		weakSelf.builds += 1;
		dispatch_semaphore_signal(entered);
		dispatch_semaphore_wait(proceed, DISPATCH_TIME_FOREVER);
		return backend;
	} forModelName:@"lazy"];

	NSUInteger requests = 4;
	NSMutableArray<NSString *> *texts = [NSMutableArray array];
	dispatch_group_t group = dispatch_group_create();
	for (NSUInteger index = 0; index < requests; index++) {
		dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			NSError *error = nil;
			NSString *text = [self run:@"lazy" on:server error:&error].text ?: error.localizedDescription ?: @"";
			@synchronized (texts) {
				[texts addObject:text];
			}
		});
	}
	XCTAssertEqual(dispatch_semaphore_wait(entered, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)), 0);
	XCTAssertEqual([server residencyForModelName:@"lazy"], NFKServerModelResidencyLoading);
	[self waitUntil:^BOOL {
		return [[NFKServerStatus fetchFromBaseURL:server.localBaseURL apiKey:nil error:NULL] modelNamed:@"lazy"].queued == (NSInteger)requests;
	} description:@"every request waits for the build"];
	dispatch_semaphore_signal(proceed);
	XCTAssertEqual(dispatch_group_wait(group, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)), 0);

	XCTAssertEqual(self.builds, 1);
	XCTAssertEqualObjects(texts, (@[ @"ok", @"ok", @"ok", @"ok" ]));
	XCTAssertEqual([server residencyForModelName:@"lazy"], NFKServerModelResidencyResident);
}

- (void)testAFailedBuildFailsItsRequestsAndTheNextRequestRetries
{
	NFKInferenceServer *server = [self startedServer:[self server]];
	NFKResidencyTestBackend *backend = [[NFKResidencyTestBackend alloc] init];
	__weak NFKServedResidencyTests *weakSelf = self;
	[server addBackendProvider:^id<NFKInferenceBackend>(NSError **error) {
		weakSelf.builds += 1;
		if (weakSelf.builds > 1) {
			return backend;
		}
		if (error != NULL) {
			*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceRefused
									 userInfo:@{ NSLocalizedDescriptionKey: @"the weights are missing" }];
		}
		return nil;
	} forModelName:@"lazy"];

	NSError *error = nil;
	XCTAssertNil([self run:@"lazy" on:server error:&error]);
	XCTAssertEqualObjects(error.domain, NFKInferenceErrorDomain);
	XCTAssertEqual(error.code, kNFKError_InferenceRefused, @"the request carries the provider's error: %@", error);
	XCTAssertEqual([server residencyForModelName:@"lazy"], NFKServerModelResidencyNotLoaded);
	NFKServerModelStatus *status = [[NFKServerStatus fetchFromBaseURL:server.localBaseURL apiKey:nil error:NULL] modelNamed:@"lazy"];
	XCTAssertEqualObjects(status.loadError, @"the weights are missing");
	XCTAssertEqual(status.failed, 1);

	XCTAssertEqualObjects([self run:@"lazy" on:server error:&error].text, @"ok", @"%@", error);
	XCTAssertEqual(self.builds, 2);
	status = [[NFKServerStatus fetchFromBaseURL:server.localBaseURL apiKey:nil error:NULL] modelNamed:@"lazy"];
	XCTAssertNil(status.loadError, @"a successful build clears the error");
	XCTAssertEqual(status.residency, NFKServerModelResidencyResident);
}

#pragma mark Unloading by hand

- (void)testAnUnloadDuringARunWaitsForTheRun
{
	NFKInferenceServer *server = [self startedServer:[self server]];
	server.idleUnloadInterval = 1;
	NFKResidencyTestBackend *backend = [[NFKResidencyTestBackend alloc] init];
	backend.gate = dispatch_semaphore_create(0);
	[server addBackendProvider:[self providerReturning:backend] forModelName:@"lazy"];

	__block NSString *text = nil;
	XCTestExpectation *answered = [self expectationWithDescription:@"the run ends"];
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		text = [self run:@"lazy" on:server error:NULL].text;
		[answered fulfill];
	});
	[self waitUntil:^BOOL {
		return backend.running == 1;
	} description:@"the run holds its slot"];

	XCTAssertTrue([server unloadModelNamed:@"lazy"]);
	self.now += 100;
	XCTAssertEqualObjects([server unloadIdleModels], @[], @"a model with a run in flight stays");
	XCTAssertEqual([server residencyForModelName:@"lazy"], NFKServerModelResidencyResident, @"the unload waits for the run");

	dispatch_semaphore_signal(backend.gate);
	[self waitForExpectations:@[ answered ] timeout:10];
	XCTAssertEqualObjects(text, @"ok", @"the run was not cut");
	[self waitUntil:^BOOL {
		return [server residencyForModelName:@"lazy"] == NFKServerModelResidencyNotLoaded;
	} description:@"the unload follows the run"];
	XCTAssertNil([server backendForModelName:@"lazy"]);
	XCTAssertEqualObjects([server modelNames], @[ @"lazy" ], @"the model stays hosted");

	backend.gate = nil;
	NSError *error = nil;
	XCTAssertEqualObjects([self run:@"lazy" on:server error:&error].text, @"ok", @"%@", error);
	XCTAssertEqual(self.builds, 2, @"the next request builds the model again");
	NFKServerModelStatus *status = [[NFKServerStatus fetchFromBaseURL:server.localBaseURL apiKey:nil error:NULL] modelNamed:@"lazy"];
	XCTAssertEqual(status.loads, 2);
	XCTAssertEqual(status.unloads, 1);
}

- (void)testAModelHostedWithABackendStaysResident
{
	NFKInferenceServer *server = [self server];
	server.idleUnloadInterval = 1;
	server.unloadsUnderMemoryPressure = YES;
	self.pressure = 1;
	[server addBackend:[[NFKResidencyTestBackend alloc] init] forModelName:@"eager"];

	XCTAssertFalse([server unloadModelNamed:@"eager"]);
	self.now += 1000;
	XCTAssertEqualObjects([server unloadIdleModels], @[]);
	XCTAssertEqual([server residencyForModelName:@"eager"], NFKServerModelResidencyResident);
	XCTAssertEqual([server idleTimeoutForModelName:@"eager"], INFINITY);
	XCTAssertFalse([server unloadModelNamed:@"absent"]);
}

#pragma mark Idle and pressure unloading

- (void)testIdleUnloadFiresAfterTheInterval
{
	NFKInferenceServer *server = [self server];
	server.idleUnloadInterval = 60;
	[server addBackendProvider:[self providerReturning:[[NFKResidencyTestBackend alloc] init]] forModelName:@"lazy"];
	[self load:@"lazy" on:server];
	XCTAssertEqual([server idleTimeoutForModelName:@"lazy"], 60);

	self.now += 59;
	XCTAssertEqualObjects([server unloadIdleModels], @[]);
	self.now += 2;
	XCTAssertEqualObjects([server unloadIdleModels], @[ @"lazy" ]);
	XCTAssertEqual([server residencyForModelName:@"lazy"], NFKServerModelResidencyNotLoaded);

	[server setIdleUnloadInterval:@5 forModelName:@"lazy"];
	XCTAssertEqual([server idleTimeoutForModelName:@"lazy"], 5, @"a model's own interval overrides the server's");
	[server setIdleUnloadInterval:@0 forModelName:@"lazy"];
	XCTAssertEqual([server idleTimeoutForModelName:@"lazy"], INFINITY, @"0 turns idle unloading off for the model");
	[server setIdleUnloadInterval:nil forModelName:@"lazy"];
	XCTAssertEqual([server idleTimeoutForModelName:@"lazy"], 60);
}

- (void)testPressureShortensTheTimeout
{
	NFKInferenceServer *server = [self startedServer:[self server]];
	server.idleUnloadInterval = 600;
	server.unloadsUnderMemoryPressure = YES;
	server.pressureUnloadHysteresis = 0;
	[server addBackendProvider:[self providerReturning:[[NFKResidencyTestBackend alloc] init]] forModelName:@"lazy"];
	[self load:@"lazy" on:server];

	self.pressure = 0.3;
	XCTAssertEqual([server idleTimeoutForModelName:@"lazy"], 600, @"below the onset the idle interval holds");
	self.pressure = 0.8;
	XCTAssertEqualWithAccuracy([server idleTimeoutForModelName:@"lazy"], 305, 1e-9, @"halfway from the onset to full");
	self.pressure = 1;
	XCTAssertEqual([server idleTimeoutForModelName:@"lazy"], 10, @"full pressure reaches the shortest interval");

	self.pressure = 0.8;
	NFKServerModelStatus *status = [[NFKServerStatus fetchFromBaseURL:server.localBaseURL apiKey:nil error:NULL] modelNamed:@"lazy"];
	XCTAssertEqualWithAccuracy(status.idleTimeoutSeconds.doubleValue, 305, 1e-9);
	XCTAssertLessThan(fabs(status.lastUsedDate.timeIntervalSinceNow), 5.0);

	self.now += 304;
	XCTAssertEqualObjects([server unloadIdleModels], @[]);
	self.now += 2;
	XCTAssertEqualObjects([server unloadIdleModels], @[ @"lazy" ]);
}

- (void)testAnOptionThatIsOffContributesNoLimit
{
	NFKInferenceServer *server = [self server];
	[server addBackendProvider:[self providerReturning:[[NFKResidencyTestBackend alloc] init]] forModelName:@"lazy"];
	[self load:@"lazy" on:server];
	self.pressure = 1;
	XCTAssertEqual([server idleTimeoutForModelName:@"lazy"], INFINITY, @"both options off");
	XCTAssertEqual(server.memoryPressure, 0, @"pressure is not read while its option is off");
	server.idleUnloadInterval = 30;
	XCTAssertEqual([server idleTimeoutForModelName:@"lazy"], 30, @"pressure off leaves the idle interval");

	server.idleUnloadInterval = 0;
	server.unloadsUnderMemoryPressure = YES;
	server.pressureUnloadHysteresis = 0;
	self.pressure = 0.5;
	XCTAssertEqual([server idleTimeoutForModelName:@"lazy"], INFINITY, @"idle off and no pressure: the model stays");
	self.pressure = 0.6;
	XCTAssertEqual([server idleTimeoutForModelName:@"lazy"], 600, @"at the onset the pressure interval applies");
	self.pressure = 1;
	XCTAssertEqual([server idleTimeoutForModelName:@"lazy"], 10);

	server.unloadsUnderMemoryPressure = NO;
	self.now += 1e6;
	XCTAssertEqualObjects([server unloadIdleModels], @[]);
	XCTAssertEqual([server residencyForModelName:@"lazy"], NFKServerModelResidencyResident);
}

- (void)testTheGracePeriodKeepsANewlyBuiltModel
{
	NFKInferenceServer *server = [self server];
	server.idleUnloadInterval = 10;
	server.unloadGracePeriod = 30;
	[server addBackendProvider:[self providerReturning:[[NFKResidencyTestBackend alloc] init]] forModelName:@"lazy"];
	[self load:@"lazy" on:server];

	self.now += 20;
	XCTAssertEqualObjects([server unloadIdleModels], @[], @"idle past its interval, but within the grace period");
	self.now += 11;
	XCTAssertEqualObjects([server unloadIdleModels], @[ @"lazy" ]);
}

- (void)testPressureUnloadsTheLeastRecentlyUsedModelFirst
{
	NFKInferenceServer *server = [self server];
	server.unloadsUnderMemoryPressure = YES;
	server.pressureUnloadHysteresis = 5;
	[server addBackendProvider:[self providerReturning:[[NFKResidencyTestBackend alloc] init]] forModelName:@"older"];
	[server addBackendProvider:[self providerReturning:[[NFKResidencyTestBackend alloc] init]] forModelName:@"newer"];
	[self load:@"older" on:server];
	self.now += 5;
	[self load:@"newer" on:server];
	self.pressure = 1;

	self.now += 100;
	XCTAssertEqualObjects([server unloadIdleModels], @[ @"older" ], @"one model per check, the least recently used");
	XCTAssertEqualObjects([server unloadIdleModels], @[], @"the next pressure unload waits the hysteresis");
	self.now += 6;
	XCTAssertEqualObjects([server unloadIdleModels], @[ @"newer" ]);
}

#pragma mark Relief

- (NFKReliefTestBackend *)reliefBackendHolding:(NSInteger)residentBytes conversations:(NSInteger)conversations
										experts:(NSInteger)experts
{
	NFKReliefTestBackend *backend = [[NFKReliefTestBackend alloc] init];
	backend.residentBytesHeld = residentBytes;
	backend.conversationBytes = conversations;
	backend.expertBytes = experts;
	return backend;
}

- (NFKMemoryRelief *)relieve:(NSInteger)byteCount on:(NFKInferenceServer *)server
{
	__block NFKMemoryRelief *answer = nil;
	XCTestExpectation *relieved = [self expectationWithDescription:@"relieved"];
	[server relieveMemoryByteCount:byteCount completionHandler:^(NFKMemoryRelief *relief) {
		answer = relief;
		[relieved fulfill];
	}];
	[self waitForExpectations:@[ relieved ] timeout:10];
	return answer;
}

- (void)testReliefClimbsTheCachesBeforeItReleasesAModel
{
	NFKInferenceServer *server = [self server];
	NFKReliefTestBackend *older = [self reliefBackendHolding:1000 conversations:100 experts:300];
	NFKReliefTestBackend *newer = [self reliefBackendHolding:2000 conversations:50 experts:0];
	[server addBackendProvider:[self providerReturning:older] forModelName:@"older"];
	[server addBackendProvider:[self providerReturning:newer] forModelName:@"newer"];
	[self load:@"older" on:server];
	self.now += 5;
	[self load:@"newer" on:server];

	NFKMemoryRelief *conversations = [self relieve:120 on:server];
	XCTAssertEqual(conversations.conversationCacheByteCount, 150, @"%@", conversations);
	XCTAssertEqual(conversations.expertCacheByteCount, 0);
	XCTAssertEqualObjects(conversations.releasedModelNames, @[]);

	NFKMemoryRelief *experts = [self relieve:200 on:server];
	XCTAssertEqual(experts.expertCacheByteCount, 200, @"%@", experts);
	XCTAssertEqual(older.expertBytes, 100, @"the cache gives up what was asked, not all of it");
	XCTAssertEqualObjects(experts.releasedModelNames, @[]);

	NFKMemoryRelief *models = [self relieve:1500 on:server];
	XCTAssertEqual(models.expertCacheByteCount, 100, @"%@", models);
	XCTAssertEqualObjects(models.releasedModelNames, (@[ @"older", @"newer" ]), @"least recently used first");
	XCTAssertEqual(models.modelByteCount, 3000);
	XCTAssertEqual(models.byteCount, 3100);
	XCTAssertEqual([server residencyForModelName:@"older"], NFKServerModelResidencyNotLoaded);
	XCTAssertEqual([server residencyForModelName:@"newer"], NFKServerModelResidencyNotLoaded);
}

- (void)testTheRuntimesCacheGoesLastAndCountsOnlyWhatItHeldBefore
{
	[NFKInferenceServer registerRuntimeStatusClassName:@"NFKReliefTestRuntime"];
	[self addTeardownBlock:^{
		[NFKReliefTestRuntime setCachedByteCount:0];
	}];
	NFKInferenceServer *server = [self server];
	NFKReliefTestBackend *backend = [self reliefBackendHolding:1000 conversations:150 experts:0];
	backend.parksReleasedCaches = YES;
	[server addBackendProvider:[self providerReturning:backend] forModelName:@"cached"];
	[self load:@"cached" on:server];
	[NFKReliefTestRuntime setCachedByteCount:500];

	NFKMemoryRelief *relief = [self relieve:600 on:server];
	XCTAssertEqual(relief.conversationCacheByteCount, 150, @"%@", relief);
	XCTAssertEqual(relief.runtimeCacheByteCount, 500, @"the conversation bytes the runtime held are counted once");
	XCTAssertEqual([NFKReliefTestRuntime cachedByteCount], 0, @"the released conversation caches reached the system");
	XCTAssertEqualObjects(relief.releasedModelNames, @[]);
}

- (void)testReliefKeepsPinnedModelsAndModelsWithinTheirGracePeriod
{
	NFKInferenceServer *server = [self server];
	server.expertCacheReliefFloor = 64;
	NFKReliefTestBackend *pinned = [self reliefBackendHolding:1000 conversations:0 experts:100];
	[server addBackendProvider:[self providerReturning:pinned] forModelName:@"pinned"];
	[server addBackendProvider:[self providerReturning:[self reliefBackendHolding:1000 conversations:0 experts:0]]
				  forModelName:@"fresh"];
	[server setPinned:YES forModelName:@"pinned"];
	[self load:@"pinned" on:server];
	server.unloadGracePeriod = 60;
	[self load:@"fresh" on:server];

	NFKMemoryRelief *relief = [self relieve:10000 on:server];
	XCTAssertEqual(relief.expertCacheByteCount, 36, @"down to the floor: %@", relief);
	XCTAssertEqual(pinned.expertBytes, 64);
	XCTAssertEqualObjects(relief.releasedModelNames, @[]);
	XCTAssertEqual([server residencyForModelName:@"pinned"], NFKServerModelResidencyResident);
	XCTAssertEqual([server residencyForModelName:@"fresh"], NFKServerModelResidencyResident);
}

- (void)testPressureReleasesCachesBeforeAModel
{
	NFKInferenceServer *server = [self server];
	server.unloadsUnderMemoryPressure = YES;
	server.pressureUnloadHysteresis = 5;
	NFKReliefTestBackend *backend = [self reliefBackendHolding:1000 conversations:1200 experts:0];
	[server addBackendProvider:[self providerReturning:backend] forModelName:@"cached"];
	[self load:@"cached" on:server];
	self.pressure = 1;

	self.now += 100;
	XCTAssertEqualObjects([server unloadIdleModels], @[], @"the conversation caches freed the model's bytes");
	XCTAssertEqual(backend.conversationBytes, 0);
	XCTAssertEqual([server residencyForModelName:@"cached"], NFKServerModelResidencyResident);
	self.now += 6;
	XCTAssertEqualObjects([server unloadIdleModels], (@[ @"cached" ]), @"with the caches empty the model goes");
}

#pragma mark Admission

/*! A budget of 10,000 bytes, whose reserve leaves 5,000 for the models. */
- (NFKModelMemoryBudget *)admissionBudget
{
	NFKModelMemoryBudget *budget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:11765];
	XCTAssertEqual(budget.byteCount, 10000);
	XCTAssertEqual(budget.reserveByteCount, 5000);
	return budget;
}

- (nullable NSError *)loadReturningError:(NSString *)modelName on:(NFKInferenceServer *)server
{
	__block NSError *failure = nil;
	XCTestExpectation *ended = [self expectationWithDescription:modelName];
	[server loadModelNamed:modelName completionHandler:^(NSError *error) {
		failure = error;
		[ended fulfill];
	}];
	[self waitForExpectations:@[ ended ] timeout:10];
	return failure;
}

/*! Hosts "held" and "next", each holding 3,000 bytes, and leaves "held" resident with "next" measured and
	unloaded, so the next build of "next" needs 3,000 bytes beside the 3,000 "held" holds. */
- (NFKInferenceServer *)serverHoldingAModelBesideAMeasuredOne
{
	NFKInferenceServer *server = [self server];
	server.admissionMemoryBudget = [self admissionBudget];
	[server addBackendProvider:[self providerReturning:[self reliefBackendHolding:3000 conversations:0 experts:0]]
				  forModelName:@"next"];
	[self load:@"next" on:server];
	XCTAssertTrue([server unloadModelNamed:@"next"]);
	[self waitUntil:^BOOL {
		return [server residencyForModelName:@"next"] == NFKServerModelResidencyNotLoaded;
	} description:@"the first build of next is unloaded"];
	[server addBackendProvider:[self providerReturning:[self reliefBackendHolding:3000 conversations:0 experts:0]]
				  forModelName:@"held"];
	[self load:@"held" on:server];
	return server;
}

- (void)testABuildThatDoesNotFitReleasesAnIdleModelFirst
{
	NFKInferenceServer *server = [self serverHoldingAModelBesideAMeasuredOne];
	self.now += 5;
	XCTAssertNil([self loadReturningError:@"next" on:server]);
	XCTAssertEqual([server residencyForModelName:@"next"], NFKServerModelResidencyResident);
	XCTAssertEqual([server residencyForModelName:@"held"], NFKServerModelResidencyNotLoaded, @"held made room");
}

- (void)testABuildThatCannotMakeRoomNamesWhatItConsidered
{
	NFKInferenceServer *server = [self serverHoldingAModelBesideAMeasuredOne];
	[server setPinned:YES forModelName:@"held"];
	NSError *refusal = [self loadReturningError:@"next" on:server];
	XCTAssertEqualObjects(refusal.domain, NFKModelCatalogErrorDomain);
	XCTAssertEqual(refusal.code, NFKModelCatalogErrorDoesNotFit);
	XCTAssertEqualObjects(refusal.userInfo[NFKInferenceServerAdmissionModelsKey],
						  (@[ @{ @"model": @"held", @"bytes": @3000, @"reason": @"pinned" } ]));
	XCTAssertTrue([refusal.localizedDescription containsString:@"held (pinned"], @"%@", refusal);
	XCTAssertEqual([server residencyForModelName:@"held"], NFKServerModelResidencyResident);

	[server setPinned:NO forModelName:@"held"];
	server.admitsModelsByMemory = NO;
	XCTAssertNil([self loadReturningError:@"next" on:server], @"without admission the build goes ahead");
	XCTAssertEqual([server residencyForModelName:@"held"], NFKServerModelResidencyResident);
}

/*! A server holding a pinned model of heldBytes beside the catalog model "staged", whose two stages hold
	3,500 bytes each and whose residency is automatic, built by builder from a catalog with a 64 GiB working
	set, and admitted against admissionBudget. */
- (NFKInferenceServer *)serverHolding:(NSInteger)heldBytes besideAStagedModelBuiltBy:(NFKAdmissionTestBuilder *)builder
{
	return [self serverHolding:heldBytes besideAStagedModelBuiltBy:builder
				 catalogBudget:[NFKModelMemoryBudget budgetWithWorkingSetByteCount:(NSInteger)64 << 30]
			   admissionBudget:[self admissionBudget]];
}

/*! serverHolding:besideAStagedModelBuiltBy: with the catalog's budget and the admission budget given. */
- (NFKInferenceServer *)serverHolding:(NSInteger)heldBytes besideAStagedModelBuiltBy:(NFKAdmissionTestBuilder *)builder
						catalogBudget:(NFKModelMemoryBudget *)catalogBudget
					  admissionBudget:(NFKModelMemoryBudget *)admissionBudget
{
	[NFKModelCatalog registerBackendBuilder:builder forEngine:NFKAdmissionTestEngine];
	[self addTeardownBlock:^{
		[NFKModelCatalog registerBackendBuilder:nil forEngine:NFKAdmissionTestEngine];
	}];
	NSString *json = @"{ \"schemaVersion\": 1, \"kind\": \"inferkit.model-catalog\", \"models\": [ {"
		"\"id\": \"staged.model\", \"family\": \"test\", \"engine\": \"admission-test-engine\","
		"\"tasks\": [\"text-generation\"], \"residencies\": [\"resident\", \"staged\"],"
		"\"options\": [{ \"key\": \"residency\", \"kind\": \"choice\", \"stage\": \"load\", \"default\": \"automatic\","
		"  \"choices\": [{ \"value\": \"automatic\" }, { \"value\": \"resident\" }, { \"value\": \"staged\" }] }],"
		"\"memoryModel\": { \"residency\": \"residency\" },"
		"\"variants\": [{ \"id\": \"default\", \"isDefault\": true, \"precision\": \"float16\","
		"  \"memoryFacts\": { \"stages\": [{ \"id\": \"encoder\", \"bytesByPrecision\": { \"checkpoint\": 3500 } },"
		"    { \"id\": \"decoder\", \"bytesByPrecision\": { \"checkpoint\": 3500 } }], \"basis\": \"measured\" } }] } ] }";
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	catalog.memoryBudget = catalogBudget;
	NSError *error = nil;
	XCTAssertTrue([catalog addDescriptorsFromManifestData:[json dataUsingEncoding:NSUTF8StringEncoding] error:&error],
				  @"%@", error);
	NFKInferenceServer *server = [self server];
	server.admissionMemoryBudget = admissionBudget;
	[server addBackendProvider:[self providerReturning:[self reliefBackendHolding:heldBytes conversations:0 experts:0]]
				  forModelName:@"held"];
	[server setPinned:YES forModelName:@"held"];
	[self load:@"held" on:server];
	XCTAssertTrue([server addModelWithIdentifier:@"staged.model" options:nil catalog:catalog forModelName:@"staged"
										   error:&error], @"%@", error);
	return server;
}

// The catalog's own budget holds the model resident; the admission budget leaves room for one stage, and
// the build plans in that room.
- (void)testAnAutomaticModelBuildsStagedInTheRoomAdmissionLeftAndNotTheCatalogsBudget
{
	NFKAdmissionTestBuilder *builder = [[NFKAdmissionTestBuilder alloc] init];
	NFKInferenceServer *server = [self serverHolding:1000 besideAStagedModelBuiltBy:builder];
	XCTAssertNil([self loadReturningError:@"staged" on:server], @"3,500 fits in the 4,000 the reserve leaves");
	XCTAssertEqualObjects(builder.residency, NFKModelResidencyStaged, @"the build is told the residency admitted");
	NFKModelMemoryBudget *room = builder.budget;
	XCTAssertGreaterThanOrEqual(room.byteCount - room.reserveByteCount, 4000, @"%@", room);
	XCTAssertLessThan(room.byteCount, 10000, @"the build plans in the room, not the whole budget: %@", room);
}

// An app's smaller budget on the model's catalog caps admission, as it caps the build's own fit check.
- (void)testACatalogsSmallerBudgetCapsAdmission
{
	NFKModelMemoryBudget *machine = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:(NSInteger)64 << 30];
	NFKAdmissionTestBuilder *staged = [[NFKAdmissionTestBuilder alloc] init];
	NFKInferenceServer *server = [self serverHolding:1000 besideAStagedModelBuiltBy:staged
									   catalogBudget:[self admissionBudget] admissionBudget:machine];
	XCTAssertNil([self loadReturningError:@"staged" on:server]);
	XCTAssertEqualObjects(staged.residency, NFKModelResidencyStaged, @"the 10,000-byte cap leaves room for one stage");

	NFKAdmissionTestBuilder *refused = [[NFKAdmissionTestBuilder alloc] init];
	server = [self serverHolding:3000 besideAStagedModelBuiltBy:refused catalogBudget:[self admissionBudget]
				 admissionBudget:machine];
	NSError *refusal = [self loadReturningError:@"staged" on:server];
	XCTAssertEqual(refusal.code, NFKModelCatalogErrorDoesNotFit, @"%@", refusal);
	XCTAssertNil(refused.residency, @"nothing was built");
}

- (void)testAPlannedResidencyIsHeldToTheBudgetsOwnReserve
{
	NFKAdmissionTestBuilder *builder = [[NFKAdmissionTestBuilder alloc] init];
	NFKInferenceServer *server = [self serverHolding:3000 besideAStagedModelBuiltBy:builder];
	NSError *refusal = [self loadReturningError:@"staged" on:server];
	XCTAssertEqual(refusal.code, NFKModelCatalogErrorDoesNotFit, @"3,500 exceeds the 2,000 the reserve leaves: %@", refusal);
	XCTAssertNil(builder.residency, @"nothing was built");
}

// A build under way holds what admission reserved for it, so a second build beside it is refused while the
// first has not yet measured its resident bytes.
- (void)testABuildUnderWayCountsWhatAdmissionReservedForIt
{
	NFKInferenceServer *server = [self server];
	server.admissionMemoryBudget = [self admissionBudget];
	dispatch_semaphore_t gate = dispatch_semaphore_create(0);
	__weak NFKServedResidencyTests *weakSelf = self;
	XCTestExpectation *building = [self expectationWithDescription:@"the second build of held waits in its provider"];
	NFKReliefTestBackend *backend = [self reliefBackendHolding:3000 conversations:0 experts:0];
	[server addBackendProvider:^id<NFKInferenceBackend>(NSError **error) {
		if (weakSelf.gatesBuilds) {
			[building fulfill];
			dispatch_semaphore_wait(gate, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC));
		}
		return backend;
	} forModelName:@"held"];
	[server addBackendProvider:[self providerReturning:[self reliefBackendHolding:3000 conversations:0 experts:0]]
				  forModelName:@"next"];
	for (NSString *name in @[ @"held", @"next" ]) {
		[self load:name on:server];
		XCTAssertTrue([server unloadModelNamed:name]);
		[self waitUntil:^BOOL {
			return [server residencyForModelName:name] == NFKServerModelResidencyNotLoaded;
		} description:[name stringByAppendingString:@" is measured and unloaded"]];
	}
	self.gatesBuilds = YES;
	XCTestExpectation *built = [self expectationWithDescription:@"held is built"];
	[server loadModelNamed:@"held" completionHandler:^(NSError *error) {
		[built fulfill];
	}];
	[self waitForExpectations:@[ building ] timeout:10];

	NSError *refusal = [self loadReturningError:@"next" on:server];
	XCTAssertEqual(refusal.code, NFKModelCatalogErrorDoesNotFit, @"%@", refusal);
	XCTAssertEqualObjects(refusal.userInfo[NFKInferenceServerAdmissionModelsKey],
						  (@[ @{ @"model": @"held", @"bytes": @3000, @"reason": @"loading" } ]));
	dispatch_semaphore_signal(gate);
	[self waitForExpectations:@[ built ] timeout:10];
}

// A model removed while its build waits for room ends the wait at once and releases nothing for it.
- (void)testRemovingAModelEndsItsAdmissionWithoutReleasingAnother
{
	NFKInferenceServer *server = [self startedServer:[self server]];
	server.admissionMemoryBudget = [self admissionBudget];
	server.admissionWaitInterval = 30;
	[server addBackendProvider:[self providerReturning:[self reliefBackendHolding:3000 conversations:0 experts:0]]
				  forModelName:@"next"];
	[self load:@"next" on:server];
	XCTAssertTrue([server unloadModelNamed:@"next"]);
	[self waitUntil:^BOOL {
		return [server residencyForModelName:@"next"] == NFKServerModelResidencyNotLoaded;
	} description:@"the first build of next is unloaded"];
	NFKReliefTestBackend *busy = [self reliefBackendHolding:3000 conversations:0 experts:0];
	busy.gate = dispatch_semaphore_create(0);
	[server addBackendProvider:[self providerReturning:busy] forModelName:@"busy"];
	XCTestExpectation *ran = [self expectationWithDescription:@"the run ends"];
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		[self run:@"busy" on:server error:NULL];
		[ran fulfill];
	});
	[self waitUntil:^BOOL {
		return busy.running == 1;
	} description:@"the run holds its slot"];

	__block NSError *failure = nil;
	XCTestExpectation *ended = [self expectationWithDescription:@"the load of next ends"];
	[server loadModelNamed:@"next" completionHandler:^(NSError *error) {
		failure = error;
		[ended fulfill];
	}];
	[self waitUntil:^BOOL {
		return [server residencyForModelName:@"next"] == NFKServerModelResidencyLoading;
	} description:@"next waits for room"];
	[server removeBackendForModelName:@"next"];
	[self waitForExpectations:@[ ended ] timeout:10];
	XCTAssertEqualObjects(failure.domain, NFKInferenceServerErrorDomain, @"%@", failure);
	XCTAssertEqual(failure.code, NFKInferenceServerErrorModelNotFound);

	NSInteger builds = self.builds;
	dispatch_semaphore_signal(busy.gate);
	[self waitForExpectations:@[ ran ] timeout:10];
	// A waiting admission retries every 0.25 s, so one still running would release busy within this second.
	NSDate *settled = [NSDate dateWithTimeIntervalSinceNow:1];
	while (settled.timeIntervalSinceNow > 0) {
		XCTAssertEqual([server residencyForModelName:@"busy"], NFKServerModelResidencyResident, @"nothing made room for next");
		XCTAssertEqual(self.builds, builds, @"next was not built");
		[NSThread sleepForTimeInterval:0.05];
	}
}

- (void)testABuildWaitsForARunningModelToBecomeReleasable
{
	NFKInferenceServer *server = [self startedServer:[self server]];
	server.admissionMemoryBudget = [self admissionBudget];
	server.admissionWaitInterval = 10;
	[server addBackendProvider:[self providerReturning:[self reliefBackendHolding:3000 conversations:0 experts:0]]
				  forModelName:@"next"];
	[self load:@"next" on:server];
	XCTAssertTrue([server unloadModelNamed:@"next"]);
	[self waitUntil:^BOOL {
		return [server residencyForModelName:@"next"] == NFKServerModelResidencyNotLoaded;
	} description:@"the first build of next is unloaded"];
	NFKReliefTestBackend *busy = [self reliefBackendHolding:3000 conversations:0 experts:0];
	busy.gate = dispatch_semaphore_create(0);
	[server addBackendProvider:[self providerReturning:busy] forModelName:@"busy"];
	XCTestExpectation *ran = [self expectationWithDescription:@"the run ends"];
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		[self run:@"busy" on:server error:NULL];
		[ran fulfill];
	});
	[self waitUntil:^BOOL {
		return busy.running == 1;
	} description:@"the run holds its slot"];

	dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.5 * NSEC_PER_SEC)), dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
		dispatch_semaphore_signal(busy.gate);
	});
	XCTAssertNil([self loadReturningError:@"next" on:server], @"the build waits for the run, then makes room");
	[self waitForExpectations:@[ ran ] timeout:10];
	XCTAssertEqual([server residencyForModelName:@"busy"], NFKServerModelResidencyNotLoaded);
	XCTAssertEqual([server residencyForModelName:@"next"], NFKServerModelResidencyResident);
}

- (void)testAFallingPressureTakesEffectAfterTheHysteresis
{
	NFKInferenceServer *server = [self server];
	server.unloadsUnderMemoryPressure = YES;
	server.pressureUnloadHysteresis = 10;
	self.pressure = 1;
	XCTAssertEqual(server.memoryPressure, 1);
	self.pressure = 0.2;
	self.now += 5;
	XCTAssertEqual(server.memoryPressure, 1, @"a fall is held");
	self.now += 6;
	XCTAssertEqualWithAccuracy(server.memoryPressure, 0.2, 1e-12);
	self.pressure = 0.9;
	XCTAssertEqualWithAccuracy(server.memoryPressure, 0.9, 1e-12, @"a rise takes effect at once");
}

#pragma mark Limits on hosted models

/*! Starts count requests for modelName on background queues; the group ends when every one has
	answered, and texts collects each reply's text or error description. */
- (void)startRequests:(NSUInteger)count for:(NSString *)modelName on:(NFKInferenceServer *)server
				group:(dispatch_group_t)group texts:(NSMutableArray<NSString *> *)texts
{
	for (NSUInteger index = 0; index < count; index++) {
		dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			NSError *error = nil;
			NSString *text = [self run:modelName on:server error:&error].text ?: error.localizedDescription ?: @"";
			@synchronized (texts) {
				[texts addObject:text];
			}
		});
	}
}

- (nullable NFKServerModelStatus *)status:(NSString *)modelName on:(NFKInferenceServer *)server
{
	return [[NFKServerStatus fetchFromBaseURL:server.localBaseURL apiKey:nil error:NULL] modelNamed:modelName];
}

- (void)testRaisingTheConcurrencyLimitStartsWaitingRuns
{
	NFKInferenceServer *server = [self startedServer:[self server]];
	NFKResidencyTestBackend *backend = [[NFKResidencyTestBackend alloc] init];
	backend.gate = dispatch_semaphore_create(0);
	[server addBackend:backend forModelName:@"gated"];

	dispatch_group_t group = dispatch_group_create();
	NSMutableArray<NSString *> *texts = [NSMutableArray array];
	[self startRequests:3 for:@"gated" on:server group:group texts:texts];
	[self waitUntil:^BOOL {
		return backend.running == 1 && [self status:@"gated" on:server].queued == 2;
	} description:@"one run holds the only slot and two wait"];

	server.maximumConcurrentRunsPerModel = 3;
	[self waitUntil:^BOOL {
		return backend.running == 3;
	} description:@"the raised limit starts the waiting runs"];
	XCTAssertEqual([self status:@"gated" on:server].limit, 3);

	for (NSUInteger index = 0; index < 3; index++) {
		dispatch_semaphore_signal(backend.gate);
	}
	XCTAssertEqual(dispatch_group_wait(group, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)), 0);
	XCTAssertEqualObjects(texts, (@[ @"ok", @"ok", @"ok" ]));
}

- (void)testLoweringTheConcurrencyLimitTakesEffectAsRunsEnd
{
	NFKInferenceServer *server = [self server];
	server.maximumConcurrentRunsPerModel = 2;
	[self startedServer:server];
	NFKResidencyTestBackend *backend = [[NFKResidencyTestBackend alloc] init];
	backend.gate = dispatch_semaphore_create(0);
	[server addBackend:backend forModelName:@"gated"];

	dispatch_group_t group = dispatch_group_create();
	NSMutableArray<NSString *> *texts = [NSMutableArray array];
	[self startRequests:2 for:@"gated" on:server group:group texts:texts];
	[self waitUntil:^BOOL {
		return backend.running == 2;
	} description:@"two runs hold the two slots"];

	server.maximumConcurrentRunsPerModel = 1;
	[self startRequests:1 for:@"gated" on:server group:group texts:texts];
	[self waitUntil:^BOOL {
		NFKServerModelStatus *status = [self status:@"gated" on:server];
		return status.queued == 1 && status.running == 2 && status.limit == 1;
	} description:@"the running runs keep their slots and the new one waits"];

	dispatch_semaphore_signal(backend.gate);
	[self waitUntil:^BOOL {
		return [self status:@"gated" on:server].completed == 1;
	} description:@"one run ends"];
	NFKServerModelStatus *status = [self status:@"gated" on:server];
	XCTAssertEqual(status.running, 1, @"the run that ended freed no slot under the lowered limit");
	XCTAssertEqual(status.queued, 1);

	dispatch_semaphore_signal(backend.gate);
	dispatch_semaphore_signal(backend.gate);
	XCTAssertEqual(dispatch_group_wait(group, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)), 0);
	XCTAssertEqualObjects(texts, (@[ @"ok", @"ok", @"ok" ]));
}

- (void)testLoweringTheQueueLimitKeepsQueuedRunsAndRefusesLaterOnes
{
	NFKInferenceServer *server = [self startedServer:[self server]];
	NFKResidencyTestBackend *backend = [[NFKResidencyTestBackend alloc] init];
	backend.gate = dispatch_semaphore_create(0);
	[server addBackend:backend forModelName:@"gated"];

	dispatch_group_t group = dispatch_group_create();
	NSMutableArray<NSString *> *texts = [NSMutableArray array];
	[self startRequests:3 for:@"gated" on:server group:group texts:texts];
	[self waitUntil:^BOOL {
		return backend.running == 1 && [self status:@"gated" on:server].queued == 2;
	} description:@"one run holds the slot and two wait"];

	server.maximumQueuedRunsPerModel = 1;
	NFKServerModelStatus *status = [self status:@"gated" on:server];
	XCTAssertEqual(status.queued, 2, @"runs already queued stay queued");
	XCTAssertEqual(status.queueLimit, 1);
	NSError *error = nil;
	XCTAssertNil([self run:@"gated" on:server error:&error]);
	XCTAssertEqualObjects(error.domain, NFKInferenceServerErrorDomain);
	XCTAssertEqual(error.code, NFKInferenceServerErrorBusy, @"%@", error);

	for (NSUInteger index = 0; index < 3; index++) {
		dispatch_semaphore_signal(backend.gate);
	}
	XCTAssertEqual(dispatch_group_wait(group, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)), 0);
	XCTAssertEqualObjects(texts, (@[ @"ok", @"ok", @"ok" ]));
	XCTAssertEqual([self status:@"gated" on:server].refused, 1);
}

#pragma mark In-process runs

- (NFKInferenceJob *)submit:(NSString *)modelName on:(NFKInferenceServer *)server
{
	return [server submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }] modelName:modelName];
}

- (void)waitForJob:(NFKInferenceJob *)job
{
	XCTestExpectation *finished = [self expectationWithDescription:@"the job ends"];
	job.completionHandler = ^(NFKInferenceJob *ended) {
		[finished fulfill];
	};
	[self waitForExpectations:@[ finished ] timeout:10];
}

- (void)testAnInProcessJobBuildsItsModelOnAServerThatNeverStarted
{
	NFKInferenceServer *server = [self server];
	[server addBackendProvider:[self providerReturning:[[NFKResidencyTestBackend alloc] init]] forModelName:@"lazy"];

	NFKInferenceJob *first = [self submit:@"lazy" on:server];
	[self waitForJob:first];
	XCTAssertEqual(first.status, NFKInferenceJobStatusSucceeded, @"%@", first.error);
	XCTAssertEqualObjects(first.result.text, @"ok");
	NFKInferenceJob *second = [self submit:@"lazy" on:server];
	[self waitForJob:second];
	XCTAssertEqualObjects(second.result.text, @"ok", @"%@", second.error);
	XCTAssertEqual(self.builds, 1, @"the second job finds the model resident");
	XCTAssertFalse(server.isRunning);
	NFKServerModelStatus *status = [[server currentStatus] modelNamed:@"lazy"];
	XCTAssertEqual(status.residency, NFKServerModelResidencyResident);
	XCTAssertEqual(status.completed, 2, @"the server counts both runs");

	NFKInferenceJob *unknown = [self submit:@"absent" on:server];
	XCTAssertEqual(unknown.status, NFKInferenceJobStatusFailed, @"a name that resolves to no model fails at once");
	XCTAssertEqualObjects(unknown.error.domain, NFKInferenceServerErrorDomain);
	XCTAssertEqual(unknown.error.code, NFKInferenceServerErrorModelNotFound);
}

- (void)testAnInProcessRunHoldsItsModelAgainstAnIdleUnload
{
	NFKInferenceServer *server = [self server];
	server.idleUnloadInterval = 1;
	NFKResidencyTestBackend *backend = [[NFKResidencyTestBackend alloc] init];
	backend.gate = dispatch_semaphore_create(0);
	[server addBackendProvider:[self providerReturning:backend] forModelName:@"lazy"];

	NFKInferenceJob *job = [self submit:@"lazy" on:server];
	[self waitUntil:^BOOL {
		return backend.running == 1;
	} description:@"the run holds its slot"];
	XCTAssertEqual(job.status, NFKInferenceJobStatusRunning, @"the job runs once its run takes the slot");
	self.now += 100;
	XCTAssertEqualObjects([server unloadIdleModels], @[], @"a model with an in-process run stays");

	dispatch_semaphore_signal(backend.gate);
	[self waitForJob:job];
	XCTAssertEqualObjects(job.result.text, @"ok", @"%@", job.error);
	self.now += 100;
	XCTAssertEqualObjects([server unloadIdleModels], @[ @"lazy" ], @"the model unloads once the run ended and the interval passed");
}

- (void)testInProcessRunsWaitForASlotAndAFullQueueFailsTheJob
{
	NFKInferenceServer *server = [self server];
	server.maximumQueuedRunsPerModel = 1;
	NFKResidencyTestBackend *backend = [[NFKResidencyTestBackend alloc] init];
	backend.gate = dispatch_semaphore_create(0);
	[server addBackend:backend forModelName:@"gated"];

	NFKInferenceJob *running = [self submit:@"gated" on:server];
	NFKInferenceJob *queued = [self submit:@"gated" on:server];
	[self waitUntil:^BOOL {
		return backend.running == 1 && [[server currentStatus] modelNamed:@"gated"].queued == 1;
	} description:@"one run holds the slot and one waits"];
	XCTAssertEqual(queued.status, NFKInferenceJobStatusQueued);
	NFKInferenceJob *refused = [self submit:@"gated" on:server];
	XCTAssertEqual(refused.status, NFKInferenceJobStatusFailed);
	XCTAssertEqual(refused.error.code, NFKInferenceServerErrorBusy, @"%@", refused.error);

	dispatch_semaphore_signal(backend.gate);
	dispatch_semaphore_signal(backend.gate);
	[self waitForJob:running];
	[self waitForJob:queued];
	XCTAssertEqualObjects(running.result.text, @"ok", @"%@", running.error);
	XCTAssertEqualObjects(queued.result.text, @"ok", @"%@", queued.error);
	XCTAssertEqual([[server currentStatus] modelNamed:@"gated"].refused, 1);
}

- (void)testCancellingAnInProcessJobCancelsItsQueuedRun
{
	NFKInferenceServer *server = [self server];
	NFKResidencyTestBackend *backend = [[NFKResidencyTestBackend alloc] init];
	backend.gate = dispatch_semaphore_create(0);
	[server addBackend:backend forModelName:@"gated"];

	NFKInferenceJob *running = [self submit:@"gated" on:server];
	NFKInferenceJob *queued = [self submit:@"gated" on:server];
	[self waitUntil:^BOOL {
		return backend.running == 1 && [[server currentStatus] modelNamed:@"gated"].queued == 1;
	} description:@"one run holds the slot and one waits"];
	[queued cancel];
	XCTAssertEqual(queued.status, NFKInferenceJobStatusCancelled);
	XCTAssertEqual([[server currentStatus] modelNamed:@"gated"].queued, 0, @"a cancelled run no longer counts as queued");

	dispatch_semaphore_signal(backend.gate);
	[self waitForJob:running];
	XCTAssertEqualObjects(running.result.text, @"ok", @"%@", running.error);
	[self waitUntil:^BOOL {
		return [[server currentStatus] modelNamed:@"gated"].cancelled == 1;
	} description:@"the cancelled run ends without running"];
	XCTAssertEqual([[server currentStatus] modelNamed:@"gated"].completed, 1);
}

- (void)testStoppingTheListenerLeavesAnInProcessRunToFinish
{
	NFKInferenceServer *server = [self startedServer:[self server]];
	NFKResidencyTestBackend *backend = [[NFKResidencyTestBackend alloc] init];
	backend.gate = dispatch_semaphore_create(0);
	[server addBackend:backend forModelName:@"gated"];

	NFKInferenceJob *job = [self submit:@"gated" on:server];
	[self waitUntil:^BOOL {
		return backend.running == 1;
	} description:@"the run holds its slot"];
	[server stop];
	dispatch_semaphore_signal(backend.gate);
	[self waitForJob:job];
	XCTAssertEqualObjects(job.result.text, @"ok", @"%@", job.error);

	NFKInferenceJob *after = [self submit:@"gated" on:server];
	backend.gate = nil;
	[self waitForJob:after];
	XCTAssertEqualObjects(after.result.text, @"ok", @"a stopped server still runs in-process jobs: %@", after.error);
}

#pragma mark Restarting

- (void)testStopKeepsHostedModelsAndTheSameServerStartsAgain
{
	NFKInferenceServer *server = [self startedServer:[self server]];
	[server addBackendProvider:[self providerReturning:[[NFKResidencyTestBackend alloc] init]] forModelName:@"lazy"];
	NSError *error = nil;
	XCTAssertEqualObjects([self run:@"lazy" on:server error:&error].text, @"ok", @"%@", error);

	[server stop];
	XCTAssertFalse(server.isRunning);
	XCTAssertNil(server.localBaseURL);
	XCTAssertEqual([server residencyForModelName:@"lazy"], NFKServerModelResidencyResident, @"stop keeps the built backend");

	server.loopbackOnly = NO;
	server.apiKey = @"restart-key";
	server.advertisesService = YES;
	server.serviceName = [@"InferKit restart " stringByAppendingString:NSUUID.UUID.UUIDString];
	XCTAssertTrue([server startWithError:&error], @"%@", error);
	XCTAssertTrue(server.isRunning);
	XCTAssertGreaterThan(server.listeningPort, 0);
	XCTAssertEqualObjects([self run:@"lazy" on:server error:&error].text, @"ok", @"%@", error);
	XCTAssertEqual(self.builds, 1, @"the restarted server serves the backend built before the stop");

	[server stop];
	server.loopbackOnly = YES;
	XCTAssertTrue([server startWithError:&error], @"%@", error);
	NFKServerModelStatus *status = [self status:@"lazy" on:server];
	XCTAssertEqual(status.residency, NFKServerModelResidencyResident);
	XCTAssertEqual(status.completed, 2, @"the counts survive the restarts");
	XCTAssertEqual(status.loads, 1);
}

#pragma mark Pinning

- (void)testAPinnedModelIsNeverUnloadedForIdlenessOrPressure
{
	NFKInferenceServer *server = [self startedServer:[self server]];
	server.idleUnloadInterval = 10;
	server.unloadsUnderMemoryPressure = YES;
	server.pressureUnloadHysteresis = 0;
	[server addBackendProvider:[self providerReturning:[[NFKResidencyTestBackend alloc] init]] forModelName:@"pinned"];
	[server setPinned:YES forModelName:@"pinned"];
	XCTAssertEqualObjects(server.pinnedModelNames, @[ @"pinned" ]);
	XCTAssertEqual([server residencyForModelName:@"pinned"], NFKServerModelResidencyNotLoaded, @"pinning builds nothing");
	[self load:@"pinned" on:server];

	self.pressure = 1;
	self.now += 1000;
	XCTAssertEqualObjects([server unloadIdleModels], @[]);
	XCTAssertEqual([server residencyForModelName:@"pinned"], NFKServerModelResidencyResident);
	XCTAssertEqual([server idleTimeoutForModelName:@"pinned"], INFINITY);
	NFKServerModelStatus *status = [self status:@"pinned" on:server];
	XCTAssertTrue(status.isPinned);
	XCTAssertNil(status.idleTimeoutSeconds);

	XCTAssertTrue([server unloadModelNamed:@"pinned"], @"an explicit unload still unloads a pinned model");
	XCTAssertEqual([server residencyForModelName:@"pinned"], NFKServerModelResidencyNotLoaded);

	[server setPinned:NO forModelName:@"pinned"];
	XCTAssertEqualObjects(server.pinnedModelNames, @[]);
	[self load:@"pinned" on:server];
	self.now += 1000;
	XCTAssertEqualObjects([server unloadIdleModels], @[ @"pinned" ]);
	XCTAssertFalse([self status:@"pinned" on:server].isPinned);
}

@end
