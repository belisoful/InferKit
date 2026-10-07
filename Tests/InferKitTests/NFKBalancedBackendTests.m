//
//  NFKBalancedBackendTests.m
//  InferKitTests
//
//  Every test balances across real servers on free loopback ports, so what is asserted is where a
//  request actually ran.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

/*! A backend that answers with its own name, and holds each run until the test opens its gate when
	gated is set. */
@interface NFKBalancedTestBackend : NSObject <NFKInferenceBackend>
@property (nonatomic, copy) NSString *name;
@property (nonatomic, assign) BOOL gated;
@property (nonatomic, strong) dispatch_semaphore_t gate;
@property (atomic, assign) NSInteger running;
@property (atomic, assign) NSInteger answered;
@property (atomic, copy, nullable) NSString *lastConversationKey;
@end

@implementation NFKBalancedTestBackend

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_gate = dispatch_semaphore_create(0);
	}
	return self;
}

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"balanced-test";
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithObject:NFKInputPrompt];
}

- (NSDictionary<NSString *, id> *)modelInfo
{
	return @{ NFKModelInfoParameterCount: @42 };
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	@synchronized (self) {
		self.running += 1;
	}
	self.lastConversationKey = request.parameters[NFKParameterConversationKey];
	if (self.gated) {
		dispatch_semaphore_wait(self.gate, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(10 * NSEC_PER_SEC)));
	}
	@synchronized (self) {
		self.running -= 1;
		self.answered += 1;
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: self.name }];
}

@end

/*! A balancer that reads one server's thermal state as critical. */
@interface NFKBalancedStrainedBackend : NFKBalancedBackend
@property (nonatomic, copy) NSURL *strainedBaseURL;
@end

@implementation NFKBalancedStrainedBackend

- (nullable NFKServerStatus *)fetchStatusFromBaseURL:(NSURL *)baseURL apiKey:(nullable NSString *)apiKey error:(NSError **)error
{
	NFKServerStatus *status = [super fetchStatusFromBaseURL:baseURL apiKey:apiKey error:error];
	if (status == nil || ![baseURL isEqual:self.strainedBaseURL]) {
		return status;
	}
	NSMutableDictionary *reply = [status.JSONObject mutableCopy];
	NSMutableDictionary *host = [reply[@"host"] mutableCopy];
	host[@"thermal_state"] = @"critical";
	reply[@"host"] = host;
	return [NFKServerStatus statusWithJSONObject:reply];
}

@end

/*! A balancer that reads one server as expected to wait 30 seconds and every server as averaging a
	one-second run. */
@interface NFKBalancedWaitingBackend : NFKBalancedBackend
@property (nonatomic, copy) NSURL *slowBaseURL;
@end

@implementation NFKBalancedWaitingBackend

- (nullable NFKServerStatus *)fetchStatusFromBaseURL:(NSURL *)baseURL apiKey:(nullable NSString *)apiKey error:(NSError **)error
{
	NFKServerStatus *status = [super fetchStatusFromBaseURL:baseURL apiKey:apiKey error:error];
	if (status == nil) {
		return nil;
	}
	NSMutableDictionary *reply = [status.JSONObject mutableCopy];
	NSMutableArray *models = [NSMutableArray array];
	for (NSDictionary *entry in reply[@"models"]) {
		NSMutableDictionary *model = [entry mutableCopy];
		NSMutableDictionary *load = [model[@"load"] mutableCopy];
		load[@"average_run_seconds"] = @1;
		load[@"estimated_wait_seconds"] = [baseURL isEqual:self.slowBaseURL] ? @30 : @0;
		model[@"load"] = load;
		[models addObject:model];
	}
	reply[@"models"] = models;
	return [NFKServerStatus statusWithJSONObject:reply];
}

@end

@interface NFKBalancedBackendTests : XCTestCase
@property (nonatomic, assign) NSUInteger savedRetryAttempts;
@end

@implementation NFKBalancedBackendTests

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

- (NFKInferenceServer *)serverHosting:(id<NFKInferenceBackend>)backend queueLimit:(NSUInteger)queueLimit
{
	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.port = 0;
	server.loopbackOnly = YES;
	server.maximumQueuedRunsPerModel = queueLimit;
	[server addBackend:backend forModelName:@"chat"];
	NSError *error = nil;
	XCTAssertTrue([server startWithError:&error], @"%@", error);
	[self addTeardownBlock:^{
		[server stop];
	}];
	return server;
}

- (NFKBalancedTestBackend *)backendNamed:(NSString *)name
{
	NFKBalancedTestBackend *backend = [[NFKBalancedTestBackend alloc] init];
	backend.name = name;
	return backend;
}

- (NFKInferenceRequest *)request
{
	return [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }];
}

- (BOOL)waitFor:(BOOL (^)(void))condition
{
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:5];
	while (!condition() && deadline.timeIntervalSinceNow > 0) {
		[NSThread sleepForTimeInterval:0.01];
	}
	return condition();
}

#pragma mark Choosing

- (void)testARequestGoesToTheServerWithNothingOutstanding
{
	NFKBalancedTestBackend *first = [self backendNamed:@"first"];
	NFKBalancedTestBackend *second = [self backendNamed:@"second"];
	first.gated = YES;
	second.gated = YES;
	NFKBalancedBackend *balancer = [NFKBalancedBackend backendWithModelName:@"chat"];
	balancer.policy = NFKBalancingPolicyFewestOutstanding;
	[balancer addServerWithBaseURL:[self serverHosting:first queueLimit:0].localBaseURL apiKey:nil];
	[balancer addServerWithBaseURL:[self serverHosting:second queueLimit:0].localBaseURL apiKey:nil];

	dispatch_group_t group = dispatch_group_create();
	dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		[balancer runInferenceForRequest:[self request] error:NULL];
	});
	XCTAssertTrue([self waitFor:^BOOL { return first.running + second.running == 1; }]);
	dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		[balancer runInferenceForRequest:[self request] error:NULL];
	});
	XCTAssertTrue([self waitFor:^BOOL { return first.running == 1 && second.running == 1; }],
				  @"the second request went to the server the balancer had sent nothing");
	dispatch_semaphore_signal(first.gate);
	dispatch_semaphore_signal(second.gate);
	XCTAssertEqual(dispatch_group_wait(group, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(10 * NSEC_PER_SEC))), 0);
}

- (void)testRoundRobinTakesEachServerInTurn
{
	NFKBalancedTestBackend *first = [self backendNamed:@"first"];
	NFKBalancedTestBackend *second = [self backendNamed:@"second"];
	NFKBalancedBackend *balancer = [NFKBalancedBackend backendWithModelName:@"chat"];
	balancer.policy = NFKBalancingPolicyRoundRobin;
	[balancer addServerWithBaseURL:[self serverHosting:first queueLimit:0].localBaseURL apiKey:nil];
	[balancer addServerWithBaseURL:[self serverHosting:second queueLimit:0].localBaseURL apiKey:nil];
	NSMutableArray<NSString *> *answers = [NSMutableArray array];
	for (NSInteger index = 0; index < 4; index++) {
		NSError *error = nil;
		[answers addObject:[balancer runInferenceForRequest:[self request] error:&error].text ?: error.localizedDescription];
	}
	XCTAssertEqualObjects(answers, (@[ @"first", @"second", @"first", @"second" ]));
}

- (void)testAStrainedServerIsChosenOnlyWhenNoOtherRemains
{
	NFKInferenceServer *hot = [self serverHosting:[self backendNamed:@"hot"] queueLimit:0];
	NFKInferenceServer *cool = [self serverHosting:[self backendNamed:@"cool"] queueLimit:0];
	NFKBalancedStrainedBackend *balancer = [[NFKBalancedStrainedBackend alloc] init];
	balancer.modelName = @"chat";
	balancer.policy = NFKBalancingPolicyRoundRobin;
	balancer.strainedBaseURL = hot.localBaseURL;
	[balancer addServerWithBaseURL:hot.localBaseURL apiKey:nil];
	[balancer addServerWithBaseURL:cool.localBaseURL apiKey:nil];
	NSError *error = nil;
	for (NSInteger index = 0; index < 3; index++) {
		XCTAssertEqualObjects([balancer runInferenceForRequest:[self request] error:&error].text, @"cool", @"%@", error);
	}
	[balancer removeServerWithBaseURL:cool.localBaseURL];
	XCTAssertEqualObjects([balancer runInferenceForRequest:[self request] error:&error].text, @"hot",
						  @"a strained server still serves when it is the last: %@", error);
}

#pragma mark Conversations

/*! A chat's request at its turn-th user message: every turn repeats the opening and the replies. */
- (NFKInferenceRequest *)chat:(NSString *)opening turn:(NSInteger)turn
{
	NSMutableArray *messages = [NSMutableArray arrayWithObject:@{ @"role": @"system", @"content": @"be brief" }];
	[messages addObject:@{ @"role": @"user", @"content": opening }];
	for (NSInteger index = 1; index < turn; index++) {
		[messages addObject:@{ @"role": @"assistant", @"content": [NSString stringWithFormat:@"reply %ld", (long)index] }];
		[messages addObject:@{ @"role": @"user", @"content": [NSString stringWithFormat:@"and %ld?", (long)index] }];
	}
	return [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: messages }];
}

- (NFKInferenceRequest *)requestInConversation:(NSString *)conversation
{
	return [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }
									   parameters:@{ NFKParameterConversationKey: conversation }];
}

- (NFKBalancedBackend *)roundRobinOver:(NSArray<NFKInferenceServer *> *)servers
{
	NFKBalancedBackend *balancer = [NFKBalancedBackend backendWithModelName:@"chat"];
	balancer.policy = NFKBalancingPolicyRoundRobin;
	for (NFKInferenceServer *server in servers) {
		[balancer addServerWithBaseURL:server.localBaseURL apiKey:nil];
	}
	return balancer;
}

- (void)testEachTurnOfAChatGoesToTheServerThatAnsweredItsFirst
{
	NFKBalancedBackend *balancer = [self roundRobinOver:@[ [self serverHosting:[self backendNamed:@"first"] queueLimit:0],
														   [self serverHosting:[self backendNamed:@"second"] queueLimit:0] ]];
	NSError *error = nil;
	NSString *home = [balancer runInferenceForRequest:[self chat:@"hello" turn:1] error:&error].text;
	XCTAssertNotNil(home, @"%@", error);
	for (NSInteger turn = 2; turn <= 4; turn++) {
		XCTAssertEqualObjects([balancer runInferenceForRequest:[self chat:@"hello" turn:turn] error:&error].text, home,
							  @"turn %ld: %@", (long)turn, error);
	}
}

- (void)testAConversationKeyNamesTheConversation
{
	NFKBalancedBackend *balancer = [self roundRobinOver:@[ [self serverHosting:[self backendNamed:@"first"] queueLimit:0],
														   [self serverHosting:[self backendNamed:@"second"] queueLimit:0] ]];
	NSError *error = nil;
	XCTAssertEqualObjects([balancer runInferenceForRequest:[self requestInConversation:@"a"] error:&error].text, @"first", @"%@", error);
	XCTAssertEqualObjects([balancer runInferenceForRequest:[self requestInConversation:@"a"] error:&error].text, @"first", @"%@", error);
	XCTAssertEqualObjects([balancer runInferenceForRequest:[self requestInConversation:@"b"] error:&error].text, @"second", @"%@", error);
	XCTAssertEqualObjects([balancer runInferenceForRequest:[self requestInConversation:@"a"] error:&error].text, @"first", @"%@", error);
}

- (void)testAStreamedTurnKeepsItsConversation
{
	NFKBalancedBackend *balancer = [self roundRobinOver:@[ [self serverHosting:[self backendNamed:@"first"] queueLimit:0],
														   [self serverHosting:[self backendNamed:@"second"] queueLimit:0] ]];
	NSError *error = nil;
	XCTAssertEqualObjects([balancer runInferenceForRequest:[self requestInConversation:@"a"] error:&error].text, @"first", @"%@", error);
	NFKInferenceJob *job = [balancer submitInferenceJobForRequest:[self requestInConversation:@"a"]];
	XCTAssertTrue([self waitFor:^BOOL { return job.status == NFKInferenceJobStatusSucceeded; }], @"%@", job.error);
	XCTAssertEqualObjects(job.result.text, @"first");
}

- (void)testAConversationMovesWhenItsServerLeavesAndStaysWhereItMoved
{
	NFKInferenceServer *first = [self serverHosting:[self backendNamed:@"first"] queueLimit:0];
	NFKInferenceServer *second = [self serverHosting:[self backendNamed:@"second"] queueLimit:0];
	NFKBalancedBackend *balancer = [self roundRobinOver:@[ first, second ]];
	NSError *error = nil;
	XCTAssertEqualObjects([balancer runInferenceForRequest:[self requestInConversation:@"a"] error:&error].text, @"first", @"%@", error);
	[balancer removeServerWithBaseURL:first.localBaseURL];
	XCTAssertEqualObjects([balancer runInferenceForRequest:[self requestInConversation:@"a"] error:&error].text, @"second", @"%@", error);
	[balancer addServerWithBaseURL:first.localBaseURL apiKey:nil];
	for (NSInteger index = 0; index < 2; index++) {
		XCTAssertEqualObjects([balancer runInferenceForRequest:[self requestInConversation:@"a"] error:&error].text, @"second",
							  @"the conversation stays where it moved: %@", error);
	}
}

- (void)testAConversationMovesWhenItsServerWouldKeepItWaitingTooLong
{
	NFKInferenceServer *slow = [self serverHosting:[self backendNamed:@"slow"] queueLimit:0];
	NFKInferenceServer *idle = [self serverHosting:[self backendNamed:@"idle"] queueLimit:0];
	NFKBalancedWaitingBackend *balancer = [[NFKBalancedWaitingBackend alloc] init];
	balancer.modelName = @"chat";
	balancer.policy = NFKBalancingPolicyRoundRobin;
	balancer.statusInterval = 0;
	[balancer addServerWithBaseURL:slow.localBaseURL apiKey:nil];
	[balancer addServerWithBaseURL:idle.localBaseURL apiKey:nil];
	NSError *error = nil;
	XCTAssertEqualObjects([balancer runInferenceForRequest:[self requestInConversation:@"a"] error:&error].text, @"slow", @"%@", error);

	balancer.slowBaseURL = slow.localBaseURL;
	balancer.conversationWaitAllowance = 60;
	XCTAssertEqualObjects([balancer runInferenceForRequest:[self requestInConversation:@"a"] error:&error].text, @"slow",
						  @"30 seconds is within the allowance: %@", error);
	balancer.conversationWaitAllowance = 10;
	XCTAssertEqualObjects([balancer runInferenceForRequest:[self requestInConversation:@"a"] error:&error].text, @"idle",
						  @"30 seconds is past the allowance: %@", error);
}

- (void)testAnIdleConversationIsForgotten
{
	NFKBalancedBackend *balancer = [self roundRobinOver:@[ [self serverHosting:[self backendNamed:@"first"] queueLimit:0],
														   [self serverHosting:[self backendNamed:@"second"] queueLimit:0] ]];
	balancer.conversationIdleInterval = 0;
	NSError *error = nil;
	XCTAssertEqualObjects([balancer runInferenceForRequest:[self requestInConversation:@"a"] error:&error].text, @"first", @"%@", error);
	XCTAssertEqualObjects([balancer runInferenceForRequest:[self requestInConversation:@"a"] error:&error].text, @"second", @"%@", error);
}

- (void)testAChatWithoutAKeyReachesItsServerUnderTheNameTheBalancerGaveIt
{
	NFKBalancedTestBackend *only = [self backendNamed:@"only"];
	NFKBalancedBackend *balancer = [self roundRobinOver:@[ [self serverHosting:only queueLimit:0] ]];
	NSError *error = nil;
	XCTAssertNotNil([balancer runInferenceForRequest:[self chat:@"hello" turn:1] error:&error], @"%@", error);
	NSString *named = only.lastConversationKey;
	XCTAssertTrue([named hasPrefix:@"messages:"], @"%@", named);
	NFKInferenceJob *job = [balancer submitInferenceJobForRequest:[self chat:@"hello" turn:2]];
	XCTAssertTrue([self waitFor:^BOOL { return job.status == NFKInferenceJobStatusSucceeded; }], @"%@", job.error);
	XCTAssertEqualObjects(only.lastConversationKey, named, @"the streamed second turn carries the same name");
	XCTAssertNotNil([balancer runInferenceForRequest:[self chat:@"goodbye" turn:1] error:&error], @"%@", error);
	XCTAssertNotEqualObjects(only.lastConversationKey, named, @"another chat is another conversation");
}

- (void)testACallersKeyReachesTheServerAsWritten
{
	NFKBalancedTestBackend *only = [self backendNamed:@"only"];
	NFKBalancedBackend *balancer = [self roundRobinOver:@[ [self serverHosting:only queueLimit:0] ]];
	NSError *error = nil;
	XCTAssertNotNil([balancer runInferenceForRequest:[self requestInConversation:@"a"] error:&error], @"%@", error);
	XCTAssertEqualObjects(only.lastConversationKey, @"a");
	XCTAssertNotNil([balancer runInferenceForRequest:[self request] error:&error], @"%@", error);
	XCTAssertNil(only.lastConversationKey, @"a request in no conversation gains no key");
}

- (void)testTheBalancerReportsItsServersAndConversations
{
	NFKInferenceServer *first = [self serverHosting:[self backendNamed:@"first"] queueLimit:0];
	NFKInferenceServer *second = [self serverHosting:[self backendNamed:@"second"] queueLimit:0];
	NFKBalancedBackend *balancer = [self roundRobinOver:@[ first, second ]];
	NSError *error = nil;
	XCTAssertEqualObjects([balancer runInferenceForRequest:[self requestInConversation:@"a"] error:&error].text, @"first", @"%@", error);
	[balancer removeServerWithBaseURL:first.localBaseURL];
	XCTAssertEqualObjects([balancer runInferenceForRequest:[self requestInConversation:@"a"] error:&error].text, @"second", @"%@", error);
	XCTAssertNotNil([balancer runInferenceForRequest:[self requestInConversation:@"b"] error:&error], @"%@", error);

	NSDictionary *status = balancer.backendStatus;
	XCTAssertEqualObjects(status[@"policy"], @"round_robin");
	XCTAssertEqualObjects(status[@"conversation_affinity"], @YES);
	XCTAssertEqualObjects(status[@"conversations"], @2);
	XCTAssertEqualObjects(status[@"conversation_moves"], @1, @"a moved when its server left; b never moved");
	NSDictionary *only = [status[@"servers"] firstObject];
	XCTAssertEqual([status[@"servers"] count], 1u);
	XCTAssertEqualObjects(only[@"base_url"], second.localBaseURL.absoluteString);
	XCTAssertEqualObjects(only[@"hosts_model"], @YES);
	XCTAssertEqualObjects(only[@"strained"], @NO);
	XCTAssertEqualObjects(only[@"outstanding"], @0);
	XCTAssertNil(only[@"resting_seconds"]);

	NFKInferenceServer *front = [[NFKInferenceServer alloc] init];
	front.port = 0;
	front.loopbackOnly = YES;
	[front addBackend:balancer forModelName:@"chat"];
	XCTAssertTrue([front startWithError:&error], @"%@", error);
	[self addTeardownBlock:^{
		[front stop];
	}];
	NFKServerStatus *served = [NFKServerStatus fetchFromBaseURL:front.localBaseURL apiKey:nil error:&error];
	XCTAssertEqualObjects([served modelNamed:@"chat"].backendStatus[@"conversation_moves"], @1, @"%@", error);
}

- (void)testWithoutAffinityAChatTakesEachServerInTurn
{
	NFKBalancedTestBackend *first = [self backendNamed:@"first"];
	NFKBalancedBackend *balancer = [self roundRobinOver:@[ [self serverHosting:first queueLimit:0],
														   [self serverHosting:[self backendNamed:@"second"] queueLimit:0] ]];
	balancer.conversationAffinity = NO;
	NSError *error = nil;
	XCTAssertEqualObjects([balancer runInferenceForRequest:[self chat:@"hello" turn:1] error:&error].text, @"first", @"%@", error);
	XCTAssertNil(first.lastConversationKey, @"without affinity the balancer names no conversation");
	XCTAssertEqualObjects([balancer runInferenceForRequest:[self chat:@"hello" turn:2] error:&error].text, @"second", @"%@", error);
}

#pragma mark Failing over

- (void)testAFullQueueFailsOverAtOnce
{
	NFKBalancedTestBackend *busy = [self backendNamed:@"busy"];
	busy.gated = YES;
	NFKInferenceServer *busyServer = [self serverHosting:busy queueLimit:1];
	NFKInferenceServer *idleServer = [self serverHosting:[self backendNamed:@"idle"] queueLimit:0];
	NFKBalancedBackend *balancer = [NFKBalancedBackend backendWithModelName:@"chat"];
	balancer.policy = NFKBalancingPolicyRoundRobin;
	[balancer addServerWithBaseURL:busyServer.localBaseURL apiKey:nil];
	[balancer addServerWithBaseURL:idleServer.localBaseURL apiKey:nil];

	// Fill the busy server behind the balancer's back: one running, one queued.
	NFKRemoteInferKitBackend *direct = [NFKRemoteInferKitBackend backendWithBaseURL:busyServer.localBaseURL];
	dispatch_group_t group = dispatch_group_create();
	for (NSInteger index = 0; index < 2; index++) {
		dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			[direct runInferenceForRequest:[self request] error:NULL];
		});
	}
	XCTAssertTrue([self waitFor:^BOOL {
		return [[NFKServerStatus fetchFromBaseURL:busyServer.localBaseURL apiKey:nil error:NULL] modelNamed:@"chat"].queued == 1;
	}]);

	NSDate *start = [NSDate date];
	NSError *error = nil;
	XCTAssertEqualObjects([balancer runInferenceForRequest:[self request] error:&error].text, @"idle", @"%@", error);
	XCTAssertLessThan(-start.timeIntervalSinceNow, 1.0, @"the refusal is not retried on the busy server");
	XCTAssertEqualObjects(balancer.lastServerBaseURL, idleServer.localBaseURL);
	XCTAssertEqual([[NFKServerStatus fetchFromBaseURL:busyServer.localBaseURL apiKey:nil error:NULL] modelNamed:@"chat"].refused, 1,
				   @"the request reached the busy server first");

	// A fresh balancer's first turn is the busy server again.
	NFKBalancedBackend *streaming = [NFKBalancedBackend backendWithModelName:@"chat"];
	streaming.policy = NFKBalancingPolicyRoundRobin;
	[streaming addServerWithBaseURL:busyServer.localBaseURL apiKey:nil];
	[streaming addServerWithBaseURL:idleServer.localBaseURL apiKey:nil];
	XCTestExpectation *streamed = [self expectationWithDescription:@"the streamed form fails over too"];
	NFKInferenceJob *job = [streaming submitInferenceJobForRequest:[self request]];
	job.completionHandler = ^(NFKInferenceJob *finished) {
		XCTAssertEqualObjects(finished.result.text, @"idle", @"%@", finished.error);
		[streamed fulfill];
	};
	[self waitForExpectations:@[ streamed ] timeout:5];

	dispatch_semaphore_signal(busy.gate);
	dispatch_semaphore_signal(busy.gate);
	XCTAssertEqual(dispatch_group_wait(group, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(10 * NSEC_PER_SEC))), 0);
}

- (void)testAnUnreachableServerSitsOut
{
	NFKBalancedBackend *balancer = [NFKBalancedBackend backendWithModelName:@"chat"];
	balancer.policy = NFKBalancingPolicyRoundRobin;
	[balancer addServerWithBaseURL:[NSURL URLWithString:@"http://127.0.0.1:9/v1"] apiKey:nil];
	[balancer addServerWithBaseURL:[self serverHosting:[self backendNamed:@"alive"] queueLimit:0].localBaseURL apiKey:nil];
	NSError *error = nil;
	for (NSInteger index = 0; index < 3; index++) {
		XCTAssertEqualObjects([balancer runInferenceForRequest:[self request] error:&error].text, @"alive", @"%@", error);
	}
	[balancer removeServerWithBaseURL:balancer.serverBaseURLs.lastObject];
	XCTAssertNil([balancer runInferenceForRequest:[self request] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceNotReady, @"no server is left to take the model");
}

- (void)testAServerWithoutTheModelIsNotACandidate
{
	NFKInferenceServer *other = [[NFKInferenceServer alloc] init];
	other.port = 0;
	other.loopbackOnly = YES;
	[other addBackend:[self backendNamed:@"other"] forModelName:@"embedder"];
	XCTAssertTrue([other startWithError:NULL]);
	[self addTeardownBlock:^{
		[other stop];
	}];
	NFKBalancedBackend *balancer = [NFKBalancedBackend backendWithModelName:@"chat"];
	[balancer addServerWithBaseURL:other.localBaseURL apiKey:nil];
	[balancer addServerWithBaseURL:[self serverHosting:[self backendNamed:@"host"] queueLimit:0].localBaseURL apiKey:nil];
	NSError *error = nil;
	XCTAssertTrue([balancer prepareWithError:&error], @"%@", error);
	XCTAssertEqualObjects(balancer.supportedInputKeys, [NSSet setWithObject:NFKInputPrompt]);
	XCTAssertEqualObjects(balancer.modelInfo[NFKModelInfoParameterCount], @42);
	for (NSInteger index = 0; index < 3; index++) {
		XCTAssertEqualObjects([balancer runInferenceForRequest:[self request] error:&error].text, @"host", @"%@", error);
	}
}

#pragma mark A balancer as a server

- (void)testAServerHostingABalancerBalancesAndSkipsItself
{
	NFKInferenceServer *back = [self serverHosting:[self backendNamed:@"back"] queueLimit:0];
	NFKBalancedBackend *balancer = [NFKBalancedBackend backendWithModelName:@"chat"];
	NFKInferenceServer *front = [self serverHosting:balancer queueLimit:0];
	[balancer addServerWithBaseURL:front.localBaseURL apiKey:nil];
	[balancer addServerWithBaseURL:back.localBaseURL apiKey:nil];

	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:front.localBaseURL];
	client.modelName = @"chat";
	NSError *error = nil;
	for (NSInteger index = 0; index < 3; index++) {
		XCTAssertEqualObjects([client runInferenceForRequest:[self request] error:&error].text, @"back",
							  @"the front never routes to itself: %@", error);
	}
	XCTAssertEqualObjects([[NFKServerStatus fetchFromBaseURL:front.localBaseURL apiKey:nil error:NULL] modelNamed:@"chat"].backendIdentifier,
						  @"inferkit-balanced");
}

#pragma mark Discovery

- (void)testDiscoveryAddsAnAdvertisingServer
{
	NSString *model = [@"found-" stringByAppendingString:[NSUUID.UUID.UUIDString substringToIndex:8]];
	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.port = 0;
	server.apiKey = @"discoverable";
	server.serviceName = [@"InferKit balanced " stringByAppendingString:model];
	[server addBackend:[self backendNamed:@"advertised"] forModelName:model];
	NSError *error = nil;
	XCTAssertTrue([server startWithError:&error], @"%@", error);
	[self addTeardownBlock:^{
		[server stop];
	}];
	NFKBalancedBackend *balancer = [NFKBalancedBackend backendWithModelName:model];
	balancer.apiKey = @"discoverable";
	[balancer startDiscoveryWithInterval:1.0];
	[self addTeardownBlock:^{
		[balancer stopDiscovery];
	}];
	XCTAssertTrue([self waitFor:^BOOL {
		return [[balancer.serverBaseURLs valueForKey:@"port"] containsObject:@(server.listeningPort)];
	}], @"the advertisement was not added");
	XCTAssertEqualObjects([balancer runInferenceForRequest:[self request] error:&error].text, @"advertised", @"%@", error);
}

@end
