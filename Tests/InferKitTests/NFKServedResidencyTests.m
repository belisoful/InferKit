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

@interface NFKServedResidencyTests : XCTestCase
@property (atomic, assign) NSTimeInterval now;
@property (atomic, assign) double pressure;
@property (atomic, assign) NSInteger builds;
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

- (void)waitUntil:(BOOL (^)(void))condition description:(NSString *)description
{
	NSPredicate *predicate = [NSPredicate predicateWithBlock:^BOOL(id object, NSDictionary *bindings) {
		return condition();
	}];
	XCTestExpectation *met = [[XCTNSPredicateExpectation alloc] initWithPredicate:predicate object:nil];
	met.expectationDescription = description;
	[self waitForExpectations:@[ met ] timeout:10];
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
