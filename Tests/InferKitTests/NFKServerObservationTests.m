//
//  NFKServerObservationTests.m
//  InferKitTests
//
//  The server observed in process: the delegate's run, model, and client events over real HTTP on a
//  loopback port, the queue they arrive on, and the status snapshot read without a request.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

/*! The server's clock, which a test replaces. */
@interface NFKInferenceServer (NFKObservationTesting)
@property (nonatomic, copy) NSTimeInterval (^uptimeProvider)(void);
@end

/*! Records every event the server reports, and whether each arrived on the expected queue. */
@interface NFKObservationRecorder : NSObject <NFKInferenceServerDelegate>
@property (nonatomic, strong) NSMutableArray *events;
@property (atomic, assign) BOOL offQueue;
@property (nonatomic, assign) void *queueKey;
- (NSArray *)eventsOfClass:(Class)eventClass;
@end

@implementation NFKObservationRecorder

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_events = [NSMutableArray array];
	}
	return self;
}

- (void)record:(id)event
{
	if (self.queueKey != NULL && dispatch_get_specific(self.queueKey) == NULL) {
		self.offQueue = YES;
	}
	@synchronized (self.events) {
		[self.events addObject:event];
	}
}

- (NSArray *)eventsOfClass:(Class)eventClass
{
	@synchronized (self.events) {
		return [self.events filteredArrayUsingPredicate:[NSPredicate predicateWithBlock:^BOOL(id event, NSDictionary *bindings) {
			return [event isKindOfClass:eventClass];
		}]];
	}
}

- (void)inferenceServer:(NFKInferenceServer *)server didStartRun:(NFKServerRunEvent *)event
{
	[self record:event];
}

- (void)inferenceServer:(NFKInferenceServer *)server didFinishRun:(NFKServerRunEvent *)event
{
	[self record:event];
}

- (void)inferenceServer:(NFKInferenceServer *)server didChangeModel:(NFKServerModelEvent *)event
{
	[self record:event];
}

- (void)inferenceServer:(NFKInferenceServer *)server didConnectClient:(NFKServerClientEvent *)event
{
	[self record:event];
}

@end

/*! A backend that answers "ok", fails when the prompt is "fail", and holds each run while its gate is
	set until the test signals it. */
@interface NFKObservedBackend : NSObject <NFKInferenceBackend>
@property (nonatomic, strong, nullable) dispatch_semaphore_t gate;
@property (atomic, assign) NSInteger running;
@end

@implementation NFKObservedBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"observed";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	self.running += 1;
	if (self.gate != nil) {
		dispatch_semaphore_wait(self.gate, DISPATCH_TIME_FOREVER);
	}
	self.running -= 1;
	if ([[request inputForKey:NFKInputPrompt] isEqual:@"fail"]) {
		if (error != NULL) {
			*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceRefused
									 userInfo:@{ NSLocalizedDescriptionKey: @"asked to fail" }];
		}
		return nil;
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"ok" }];
}

@end

@interface NFKServerObservationTests : XCTestCase
@property (nonatomic, strong) NFKInferenceServer *server;
@property (nonatomic, strong) NFKObservationRecorder *recorder;
@property (atomic, assign) NSTimeInterval now;
@property (nonatomic, assign) NSUInteger savedRetryAttempts;
@end

@implementation NFKServerObservationTests

- (void)setUp
{
	[super setUp];
	self.savedRetryAttempts = NFKRemoteTransport.retryAttempts;
	NFKRemoteTransport.retryAttempts = 0;
	self.now = 1000;
	self.recorder = [[NFKObservationRecorder alloc] init];
	self.server = [[NFKInferenceServer alloc] init];
	self.server.port = 0;
	self.server.loopbackOnly = YES;
	self.server.residencyCheckInterval = 0;
	self.server.unloadGracePeriod = 0;
	self.server.delegate = self.recorder;
	NSError *error = nil;
	XCTAssertTrue([self.server startWithError:&error], @"%@", error);
}

- (void)tearDown
{
	[self.server stop];
	NFKRemoteTransport.retryAttempts = self.savedRetryAttempts;
	[super tearDown];
}

#pragma mark Helpers

- (nullable NFKInferenceResult *)run:(NSString *)prompt on:(NSString *)modelName error:(NSError **)error
{
	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:self.server.localBaseURL];
	client.modelName = modelName;
	return [client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: prompt }] error:error];
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

- (NSArray<NFKServerRunEvent *> *)runEventsWaitingFor:(NSUInteger)count
{
	[self waitUntil:^BOOL {
		return [self.recorder eventsOfClass:NFKServerRunEvent.class].count >= count;
	} description:[NSString stringWithFormat:@"%lu run events", (unsigned long)count]];
	return [self.recorder eventsOfClass:NFKServerRunEvent.class];
}

- (NSArray<NFKServerModelEvent *> *)modelEventsWaitingFor:(NSUInteger)count
{
	[self waitUntil:^BOOL {
		return [self.recorder eventsOfClass:NFKServerModelEvent.class].count >= count;
	} description:[NSString stringWithFormat:@"%lu model events", (unsigned long)count]];
	return [self.recorder eventsOfClass:NFKServerModelEvent.class];
}

#pragma mark Runs

- (void)testARunIsReportedWhenItStartsAndWhenItEnds
{
	[self.server addBackend:[[NFKObservedBackend alloc] init] forModelName:@"observed"];
	NSError *error = nil;
	XCTAssertEqualObjects([self run:@"hello" on:@"observed" error:&error].text, @"ok", @"%@", error);
	XCTAssertNil([self run:@"fail" on:@"observed" error:&error]);

	NSArray<NFKServerRunEvent *> *events = [self runEventsWaitingFor:4];
	XCTAssertEqual(events.count, 4);
	NFKServerRunEvent *started = events[0];
	NFKServerRunEvent *completed = events[1];
	XCTAssertEqual(started.state, NFKServerRunStateStarted);
	XCTAssertEqual(completed.state, NFKServerRunStateCompleted);
	XCTAssertEqualObjects(started.identifier, completed.identifier, @"a run's two events share its identifier");
	XCTAssertEqualObjects(completed.modelName, @"observed");
	XCTAssertEqualObjects(completed.route, @"/v1/inferkit/run");
	NSArray<NSString *> *loopbackAddresses = @[ @"127.0.0.1", @"::1" ];
	XCTAssertTrue([loopbackAddresses containsObject:completed.clientAddress], @"%@", completed.clientAddress);
	XCTAssertNil(completed.keyName);
	XCTAssertNil(completed.error);
	XCTAssertGreaterThanOrEqual(completed.durationSeconds, 0);
	XCTAssertEqual(started.durationSeconds, 0);

	NFKServerRunEvent *failed = events[3];
	XCTAssertEqual(failed.state, NFKServerRunStateFailed);
	XCTAssertEqual(failed.error.code, kNFKError_InferenceRefused);
	XCTAssertNotEqualObjects(failed.identifier, completed.identifier);
}

- (void)testARefusedRunIsReportedOnceWithoutAStart
{
	self.server.maximumQueuedRunsPerModel = 1;
	NFKObservedBackend *backend = [[NFKObservedBackend alloc] init];
	backend.gate = dispatch_semaphore_create(0);
	[self.server addBackend:backend forModelName:@"gated"];

	dispatch_group_t group = dispatch_group_create();
	for (NSUInteger index = 0; index < 2; index++) {
		dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			[self run:@"hello" on:@"gated" error:NULL];
		});
	}
	[self waitUntil:^BOOL {
		return backend.running == 1 && [[self.server currentStatus] modelNamed:@"gated"].queued == 1;
	} description:@"one run holds the slot and one waits"];
	NSError *error = nil;
	XCTAssertNil([self run:@"hello" on:@"gated" error:&error]);
	XCTAssertEqual(error.code, NFKInferenceServerErrorBusy);

	NSArray<NFKServerRunEvent *> *events = [self runEventsWaitingFor:2];
	NFKServerRunEvent *refused = events.lastObject;
	XCTAssertEqual(refused.state, NFKServerRunStateRefused);
	XCTAssertEqual(refused.durationSeconds, 0);
	XCTAssertEqual(refused.error.code, NFKInferenceServerErrorBusy);
	for (NFKServerRunEvent *event in events) {
		BOOL startOfRefused = event.state == NFKServerRunStateStarted && [event.identifier isEqualToString:refused.identifier];
		XCTAssertFalse(startOfRefused, @"a refused run never starts");
	}

	dispatch_semaphore_signal(backend.gate);
	dispatch_semaphore_signal(backend.gate);
	XCTAssertEqual(dispatch_group_wait(group, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)), 0);
}

#pragma mark Models

- (void)testModelEventsCarryTheirReasons
{
	__weak NFKServerObservationTests *weakSelf = self;
	self.server.uptimeProvider = ^NSTimeInterval {
		return weakSelf.now;
	};
	self.server.idleUnloadInterval = 60;
	[self.server addBackendProvider:^id<NFKInferenceBackend>(NSError **error) {
		return [[NFKObservedBackend alloc] init];
	} forModelName:@"lazy"];
	[self.server addBackendProvider:^id<NFKInferenceBackend>(NSError **error) {
		if (error != NULL) {
			*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceRefused
									 userInfo:@{ NSLocalizedDescriptionKey: @"the weights are missing" }];
		}
		return nil;
	} forModelName:@"broken"];

	NSError *error = nil;
	XCTAssertEqualObjects([self run:@"hello" on:@"lazy" error:&error].text, @"ok", @"%@", error);
	XCTAssertTrue([self.server unloadModelNamed:@"lazy"]);
	XCTestExpectation *loaded = [self expectationWithDescription:@"loaded"];
	[self.server loadModelNamed:@"lazy" completionHandler:^(NSError *loadError) {
		[loaded fulfill];
	}];
	[self waitForExpectations:@[ loaded ] timeout:10];
	self.now += 61;
	XCTAssertEqualObjects([self.server unloadIdleModels], @[ @"lazy" ]);
	XCTAssertNil([self run:@"hello" on:@"broken" error:&error]);
	XCTAssertEqualObjects([self run:@"hello" on:@"lazy" error:&error].text, @"ok", @"%@", error);
	[self.server removeBackendForModelName:@"lazy"];

	NSArray<NFKServerModelEvent *> *events = [self modelEventsWaitingFor:7];
	NSMutableArray<NSString *> *summary = [NSMutableArray array];
	for (NFKServerModelEvent *event in events) {
		[summary addObject:[NSString stringWithFormat:@"%@ %ld %ld", event.modelName, (long)event.kind, (long)event.reason]];
	}
	NSArray<NSString *> *expected = @[
		[NSString stringWithFormat:@"lazy %ld %ld", (long)NFKServerModelEventKindLoaded, (long)NFKServerModelEventReasonRequest],
		[NSString stringWithFormat:@"lazy %ld %ld", (long)NFKServerModelEventKindUnloaded, (long)NFKServerModelEventReasonExplicit],
		[NSString stringWithFormat:@"lazy %ld %ld", (long)NFKServerModelEventKindLoaded, (long)NFKServerModelEventReasonExplicit],
		[NSString stringWithFormat:@"lazy %ld %ld", (long)NFKServerModelEventKindUnloaded, (long)NFKServerModelEventReasonIdle],
		[NSString stringWithFormat:@"broken %ld %ld", (long)NFKServerModelEventKindLoadFailed, (long)NFKServerModelEventReasonRequest],
		[NSString stringWithFormat:@"lazy %ld %ld", (long)NFKServerModelEventKindLoaded, (long)NFKServerModelEventReasonRequest],
		[NSString stringWithFormat:@"lazy %ld %ld", (long)NFKServerModelEventKindUnloaded, (long)NFKServerModelEventReasonRemoved],
	];
	XCTAssertEqualObjects(summary, expected);
	XCTAssertEqualObjects(events[4].error.localizedDescription, @"the weights are missing");
}

- (void)testPressureUnloadsAreReportedAsPressure
{
	__weak NFKServerObservationTests *weakSelf = self;
	self.server.uptimeProvider = ^NSTimeInterval {
		return weakSelf.now;
	};
	self.server.unloadsUnderMemoryPressure = YES;
	self.server.memoryPressureProvider = ^double {
		return 1;
	};
	[self.server addBackendProvider:^id<NFKInferenceBackend>(NSError **error) {
		return [[NFKObservedBackend alloc] init];
	} forModelName:@"lazy"];
	XCTestExpectation *loaded = [self expectationWithDescription:@"loaded"];
	[self.server loadModelNamed:@"lazy" completionHandler:^(NSError *loadError) {
		[loaded fulfill];
	}];
	[self waitForExpectations:@[ loaded ] timeout:10];
	self.now += 11;
	XCTAssertEqualObjects([self.server unloadIdleModels], @[ @"lazy" ]);

	NSArray<NFKServerModelEvent *> *events = [self modelEventsWaitingFor:2];
	XCTAssertEqual(events[1].kind, NFKServerModelEventKindUnloaded);
	XCTAssertEqual(events[1].reason, NFKServerModelEventReasonPressure);
}

#pragma mark Clients

- (void)testAClientIsReportedOncePerConnectionWithItsKey
{
	self.server.requiresAPIKeyOnLoopback = YES;
	[self.server addAPIKey:@"key-dashboard" named:@"dashboard" scopes:NFKServerKeyScopeStatus];
	NSURLSession *session = [NSURLSession sessionWithConfiguration:NSURLSessionConfiguration.ephemeralSessionConfiguration];
	NSURL *status = [self.server.localBaseURL URLByAppendingPathComponent:@"inferkit/status"];
	for (NSUInteger index = 0; index < 2; index++) {
		NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:status];
		[request setValue:@"Bearer key-dashboard" forHTTPHeaderField:@"Authorization"];
		NSHTTPURLResponse *response = nil;
		[NFKRemoteTransport sendRequest:request session:session response:&response error:NULL];
		XCTAssertEqual(response.statusCode, 200);
	}
	[session finishTasksAndInvalidate];

	[self waitUntil:^BOOL {
		return [self.recorder eventsOfClass:NFKServerClientEvent.class].count >= 1;
	} description:@"the client is reported"];
	[NSThread sleepForTimeInterval:0.2];
	NSArray<NFKServerClientEvent *> *clients = [self.recorder eventsOfClass:NFKServerClientEvent.class];
	XCTAssertEqual(clients.count, 1, @"the second request reused the connection");
	XCTAssertEqualObjects(clients.firstObject.keyName, @"dashboard");
	XCTAssertEqual(clients.firstObject.scope, NFKServerKeyScopeStatus);
	XCTAssertTrue(clients.firstObject.fromLoopback);
	XCTAssertEqualObjects(clients.firstObject.path, @"/v1/inferkit/status");
}

#pragma mark Delivery

- (void)testEventsArriveOnTheDelegateQueue
{
	static char queueKey;
	dispatch_queue_t queue = dispatch_queue_create("NFKServerObservationTests.delegate", DISPATCH_QUEUE_SERIAL);
	dispatch_queue_set_specific(queue, &queueKey, &queueKey, NULL);
	self.recorder.queueKey = &queueKey;
	self.server.delegateQueue = queue;
	XCTAssertEqual(self.server.delegateQueue, queue);
	[self.server addBackend:[[NFKObservedBackend alloc] init] forModelName:@"observed"];
	NSError *error = nil;
	XCTAssertNotNil([self run:@"hello" on:@"observed" error:&error], @"%@", error);
	[self runEventsWaitingFor:2];
	XCTAssertFalse(self.recorder.offQueue);

	self.server.delegateQueue = nil;
	XCTAssertNotNil(self.server.delegateQueue, @"nil restores the server's own queue");
	XCTAssertNotEqual(self.server.delegateQueue, queue);
}

#pragma mark Status

- (void)testTheStatusSnapshotIsTheStatusRoutesReading
{
	[self.server addBackend:[[NFKObservedBackend alloc] init] forModelName:@"observed"];
	[self.server addBackendProvider:^id<NFKInferenceBackend>(NSError **error) {
		return [[NFKObservedBackend alloc] init];
	} forModelName:@"lazy"];
	[self.server setPinned:YES forModelName:@"observed"];
	NSError *error = nil;
	XCTAssertNotNil([self run:@"hello" on:@"observed" error:&error], @"%@", error);

	NFKServerStatus *snapshot = [self.server currentStatus];
	NFKServerStatus *fetched = [NFKServerStatus fetchFromBaseURL:self.server.localBaseURL apiKey:nil error:&error];
	XCTAssertNotNil(fetched, @"%@", error);
	XCTAssertEqualObjects([snapshot.models valueForKey:@"name"], [fetched.models valueForKey:@"name"]);
	XCTAssertEqual([snapshot modelNamed:@"observed"].completed, 1);
	XCTAssertTrue([snapshot modelNamed:@"observed"].isPinned);
	XCTAssertEqual([snapshot modelNamed:@"lazy"].residency, NFKServerModelResidencyNotLoaded);
	XCTAssertEqualObjects(snapshot.version, fetched.version);
	XCTAssertNotNil(snapshot.host);

	self.server.reportsHostDetails = NO;
	XCTAssertNil([self.server currentStatus].host);
}

@end
