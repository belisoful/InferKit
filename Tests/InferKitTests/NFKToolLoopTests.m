//
//  NFKToolLoopTests.m
//  InferKitTests
//
//  The tool loop against a scripted model: the turns it sends, the tools it runs, where it stops, its
//  job's cancellation, and a hosted loop's result crossing the native route.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

/*! A model that answers each turn with the next staged reply, recording what it was sent. */
@interface NFKScriptedModel : NSObject <NFKInferenceBackend>
@property (nonatomic, strong) NSMutableArray *staged;
@property (nonatomic, strong) NSMutableArray<NFKInferenceRequest *> *requests;
@end

@implementation NFKScriptedModel

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_staged = [NSMutableArray array];
		_requests = [NSMutableArray array];
	}
	return self;
}

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"scripted";
}

- (void)stageCalls:(NSArray<NSDictionary *> *)calls usage:(nullable NSDictionary *)usage
{
	NSMutableDictionary *outputs = [NSMutableDictionary dictionaryWithObject:calls forKey:NFKOutputToolCalls];
	outputs[NFKOutputUsage] = usage;
	[self.staged addObject:[NFKInferenceResult resultWithOutputs:outputs]];
}

- (void)stageText:(NSString *)text usage:(nullable NSDictionary *)usage
{
	NSMutableDictionary *outputs = [NSMutableDictionary dictionaryWithObject:text forKey:NFKOutputText];
	outputs[NFKOutputUsage] = usage;
	[self.staged addObject:[NFKInferenceResult resultWithOutputs:outputs]];
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	@synchronized (self) {
		[self.requests addObject:request];
		id next = self.staged.firstObject;
		if (next == nil) {
			*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceBackendFailure
									 userInfo:@{ NSLocalizedDescriptionKey: @"nothing staged" }];
			return nil;
		}
		[self.staged removeObjectAtIndex:0];
		if ([next isKindOfClass:NSError.class]) {
			*error = next;
			return nil;
		}
		return next;
	}
}

@end

/*! A model whose turn never finishes until it is cancelled. */
@interface NFKHangingModel : NFKScriptedModel
@property (nonatomic, strong) XCTestExpectation *turnCancelled;
@end

@implementation NFKHangingModel

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	XCTestExpectation *cancelled = self.turnCancelled;
	job.cancellationHandler = ^{
		[cancelled fulfill];
	};
	// After the loop has installed its progress handler, which happens once this returns.
	dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.05 * NSEC_PER_SEC)), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		[job reportProgress:0.5 partialResult:[NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"Thinking" }]];
	});
	return job;
}

@end

@interface NFKToolLoopTests : XCTestCase
@property (nonatomic, strong) NFKScriptedModel *model;
@property (nonatomic, strong) NFKInferenceTool *weather;
@property (atomic, assign) NSInteger weatherCalls;
@end

@implementation NFKToolLoopTests

- (void)setUp
{
	[super setUp];
	self.model = [[NFKScriptedModel alloc] init];
	__weak NFKToolLoopTests *weakSelf = self;
	self.weather = [NFKInferenceTool toolWithName:@"get_weather" description:@"Current weather in a city."
									   parameters:@{ @"type": @"object", @"properties": @{ @"city": @{ @"type": @"string" } }, @"required": @[ @"city" ] }
										  handler:^NFKToolResult *(NSDictionary *arguments, NSError **error) {
		weakSelf.weatherCalls += 1;
		return [NFKToolResult resultWithText:[NSString stringWithFormat:@"18 C and clear in %@", arguments[@"city"]]
									 outputs:@{ @"celsius": @18 } imagePNGs:nil];
	}];
}

- (NSDictionary *)call:(NSString *)identifier city:(nullable NSString *)city
{
	NSDictionary *arguments = city != nil ? @{ @"city": city } : @{};
	return @{ @"id": identifier, @"name": @"get_weather", @"arguments": arguments };
}

- (NFKInferenceRequest *)prompt:(NSString *)text
{
	return [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: text } parameters:@{ NFKParameterMaxTokens: @64 }
								   outputModality:NFKModalityText];
}

#pragma mark The loop

- (void)testTheModelCallsAToolAndAnswersFromItsResult
{
	[self.model stageCalls:@[ [self call:@"call_1" city:@"Paris"] ] usage:nil];
	[self.model stageText:@"It is 18 C and clear in Paris." usage:nil];
	NFKToolLoop *loop = [[NFKToolLoop alloc] initWithBackend:self.model tools:@[ self.weather ]];
	NSMutableArray<NFKToolInvocation *> *seen = [NSMutableArray array];
	loop.invocationHandler = ^(NFKToolInvocation *invocation) {
		[seen addObject:invocation];
	};

	NSError *error = nil;
	NFKInferenceResult *result = [loop runInferenceForRequest:[self prompt:@"Weather in Paris?"] error:&error];
	XCTAssertNotNil(result, @"%@", error);
	XCTAssertEqualObjects(result.text, @"It is 18 C and clear in Paris.");
	XCTAssertEqual(self.weatherCalls, 1);

	NFKInferenceRequest *first = self.model.requests[0];
	XCTAssertNil(first.prompt, @"the prompt became the conversation's first turn");
	NSArray *opening = @[ @{ @"role": @"user", @"content": @"Weather in Paris?" } ];
	XCTAssertEqualObjects(first.messages, opening);
	XCTAssertEqualObjects(first.parameters[NFKParameterTools], @[ self.weather.declaration ]);
	XCTAssertEqualObjects(first.parameters[NFKParameterMaxTokens], @64, @"the caller's parameters ride on every turn");
	XCTAssertEqual(first.outputModality, NFKModalityText);

	NSArray<NSDictionary *> *second = self.model.requests[1].messages;
	XCTAssertEqual(second.count, 3u);
	XCTAssertEqualObjects(second[1][@"tool_calls"][0][@"id"], @"call_1");
	XCTAssertEqualObjects(second[2][@"role"], @"tool");
	XCTAssertEqualObjects(second[2][@"tool_call_id"], @"call_1");
	XCTAssertEqualObjects(second[2][@"content"], @"18 C and clear in Paris");

	NSArray<NFKToolInvocation *> *invocations = [result outputForKey:NFKOutputToolInvocations];
	XCTAssertEqual(invocations.count, 1u);
	XCTAssertEqualObjects(invocations[0].toolName, @"get_weather");
	XCTAssertEqualObjects(invocations[0].callIdentifier, @"call_1");
	XCTAssertEqualObjects(invocations[0].result.outputs[@"celsius"], @18, @"what the tool made for the application reaches the caller");
	XCTAssertEqualObjects(seen, invocations);

	NSArray *conversation = [result outputForKey:NFKOutputMessages];
	XCTAssertEqual(conversation.count, 4u);
	XCTAssertEqualObjects(conversation.lastObject[@"content"], @"It is 18 C and clear in Paris.");
	XCTAssertNil([result outputForKey:NFKOutputToolTurnLimitReached]);
}

- (void)testACallToAToolTheLoopDoesNotHoldIsReturnedForTheCaller
{
	NSDictionary *own = @{ @"name": @"open_door", @"description": @"Opens the door.", @"parameters": @{ @"type": @"object" } };
	[self.model stageCalls:@[ @{ @"id": @"c", @"name": @"open_door", @"arguments": @{} } ] usage:nil];
	NFKToolLoop *loop = [[NFKToolLoop alloc] initWithBackend:self.model tools:@[ self.weather ]];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Let me in." }
															   parameters:@{ NFKParameterTools: @[ own, self.weather.declaration ] }
														   outputModality:NFKModalityText];
	NFKInferenceResult *result = [loop runInferenceForRequest:request error:NULL];
	NSArray *declared = self.model.requests[0].parameters[NFKParameterTools];
	NSArray *expected = @[ self.weather.declaration, own ];
	XCTAssertEqualObjects(declared, expected, @"the caller's own tool is offered beside the loop's, without a duplicate");
	XCTAssertEqualObjects(result.toolCalls[0][@"name"], @"open_door");
	XCTAssertNil([result outputForKey:NFKOutputToolInvocations]);
	XCTAssertEqual([[result outputForKey:NFKOutputMessages] count], 2u, @"the caller answers the call and continues the conversation");
}

- (void)testTheTurnLimitReturnsTheLastCallsUnrun
{
	for (NSInteger turn = 0; turn < 3; turn++) {
		[self.model stageCalls:@[ [self call:[NSString stringWithFormat:@"c%ld", (long)turn] city:@"Rome"] ] usage:nil];
	}
	NFKToolLoop *loop = [[NFKToolLoop alloc] initWithBackend:self.model tools:@[ self.weather ]];
	loop.maximumTurns = 2;
	NFKInferenceResult *result = [loop runInferenceForRequest:[self prompt:@"Again and again."] error:NULL];
	XCTAssertEqualObjects([result outputForKey:NFKOutputToolTurnLimitReached], @YES);
	XCTAssertEqual(self.model.requests.count, 2u);
	XCTAssertEqual(self.weatherCalls, 1, @"the second turn's call is returned, not run");
	XCTAssertEqualObjects(result.toolCalls[0][@"id"], @"c1");
}

- (void)testADeclinedCallReachesTheModelAsAnErrorResult
{
	[self.model stageCalls:@[ [self call:@"c" city:@"Paris"] ] usage:nil];
	[self.model stageText:@"I was not allowed to check." usage:nil];
	NFKToolLoop *loop = [[NFKToolLoop alloc] initWithBackend:self.model tools:@[ self.weather ]];
	__block NSString *asked = nil;
	loop.approvalHandler = ^BOOL(NSString *toolName, NSDictionary *arguments) {
		asked = [NSString stringWithFormat:@"%@ %@", toolName, arguments[@"city"]];
		return NO;
	};
	NFKInferenceResult *result = [loop runInferenceForRequest:[self prompt:@"Weather?"] error:NULL];
	XCTAssertEqualObjects(asked, @"get_weather Paris");
	XCTAssertEqual(self.weatherCalls, 0);
	NSDictionary *answer = self.model.requests[1].messages[2];
	XCTAssertEqualObjects(answer[NFKMessageMetadataKey][@"isError"], @YES);
	XCTAssertTrue([answer[@"content"] containsString:@"declined"]);
	NFKToolInvocation *declined = [[result outputForKey:NFKOutputToolInvocations] firstObject];
	XCTAssertTrue(declined.result.isError);
}

- (void)testAFailingCallLetsTheModelCorrectIt
{
	[self.model stageCalls:@[ [self call:@"bad" city:nil] ] usage:nil];
	[self.model stageCalls:@[ [self call:@"good" city:@"Lima"] ] usage:nil];
	[self.model stageText:@"Lima is clear." usage:nil];
	NFKToolLoop *loop = [[NFKToolLoop alloc] initWithBackend:self.model tools:@[ self.weather ]];
	NSError *error = nil;
	NFKInferenceResult *result = [loop runInferenceForRequest:[self prompt:@"Weather?"] error:&error];
	XCTAssertEqualObjects(result.text, @"Lima is clear.", @"%@", error);
	NSArray<NFKToolInvocation *> *invocations = [result outputForKey:NFKOutputToolInvocations];
	XCTAssertTrue(invocations[0].result.isError);
	XCTAssertTrue([invocations[0].result.text containsString:@"\"city\" is missing"]);
	XCTAssertFalse(invocations[1].result.isError);
	XCTAssertEqual(self.weatherCalls, 1);
}

- (void)testAFailingModelTurnEndsTheLoopWithItsError
{
	[self.model stageCalls:@[ [self call:@"c" city:@"Paris"] ] usage:nil];
	[self.model.staged addObject:[NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceRateLimited userInfo:nil]];
	NFKToolLoop *loop = [[NFKToolLoop alloc] initWithBackend:self.model tools:@[ self.weather ]];
	NSError *error = nil;
	XCTAssertNil([loop runInferenceForRequest:[self prompt:@"Weather?"] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceRateLimited);
}

- (void)testUsageIsSummedOverTheTurns
{
	[self.model stageCalls:@[ [self call:@"c" city:@"Paris"] ] usage:@{ NFKUsageInputTokens: @100, NFKUsageOutputTokens: @20 }];
	[self.model stageText:@"Clear." usage:@{ NFKUsageInputTokens: @150, NFKUsageOutputTokens: @5, NFKUsageCachedTokens: @90 }];
	NFKToolLoop *loop = [[NFKToolLoop alloc] initWithBackend:self.model tools:@[ self.weather ]];
	NSDictionary *usage = [[loop runInferenceForRequest:[self prompt:@"Weather?"] error:NULL] outputForKey:NFKOutputUsage];
	NSDictionary *expected = @{ NFKUsageInputTokens: @250, NFKUsageOutputTokens: @25, NFKUsageCachedTokens: @90 };
	XCTAssertEqualObjects(usage, expected);
}

- (void)testARequestWithoutTextIsRefused
{
	NFKToolLoop *loop = [[NFKToolLoop alloc] initWithBackend:self.model tools:@[ self.weather ]];
	NSError *error = nil;
	XCTAssertNil([loop runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);
	XCTAssertEqualObjects(loop.backendIdentifier, @"tool-loop");
	XCTAssertTrue([loop.supportedParameterKeys containsObject:NFKParameterTools]);
	XCTAssertEqual([loop toolNamed:@"get_weather"], self.weather);
}

#pragma mark The job

- (void)testTheJobFinishesWithTheLoopsResult
{
	[self.model stageCalls:@[ [self call:@"c" city:@"Paris"] ] usage:nil];
	[self.model stageText:@"Clear." usage:nil];
	NFKToolLoop *loop = [[NFKToolLoop alloc] initWithBackend:self.model tools:@[ self.weather ]];
	NFKInferenceJob *job = [loop submitInferenceJobForRequest:[self prompt:@"Weather?"]];
	XCTestExpectation *finished = [self expectationWithDescription:@"finished"];
	job.completionHandler = ^(NFKInferenceJob *completed) {
		[finished fulfill];
	};
	[self waitForExpectations:@[ finished ] timeout:10];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	XCTAssertEqualObjects(job.result.text, @"Clear.");
}

- (void)testCancellingTheJobCancelsTheModelsTurnAndReportsItsPartials
{
	NFKHangingModel *model = [[NFKHangingModel alloc] init];
	model.turnCancelled = [self expectationWithDescription:@"turn cancelled"];
	NFKToolLoop *loop = [[NFKToolLoop alloc] initWithBackend:model tools:@[ self.weather ]];
	NFKInferenceJob *job = [loop submitInferenceJobForRequest:[self prompt:@"Weather?"]];
	XCTestExpectation *partial = [self expectationWithDescription:@"partial"];
	partial.assertForOverFulfill = NO;
	job.progressHandler = ^(NFKInferenceJob *reporting) {
		if ([reporting.partialResult.text isEqualToString:@"Thinking"]) {
			[partial fulfill];
		}
	};
	[self waitForExpectations:@[ partial ] timeout:10];
	[job cancel];
	[self waitForExpectations:@[ model.turnCancelled ] timeout:10];
	XCTAssertEqual(job.status, NFKInferenceJobStatusCancelled);
}

#pragma mark Serving

- (void)testAHostedLoopsInvocationsCrossTheNativeRoute
{
	[self.model stageCalls:@[ [self call:@"c" city:@"Paris"] ] usage:nil];
	[self.model stageText:@"Clear." usage:nil];
	NFKToolLoop *loop = [[NFKToolLoop alloc] initWithBackend:self.model tools:@[ self.weather ]];
	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.port = 0;
	server.loopbackOnly = YES;
	[server addBackend:loop forModelName:@"weather-agent"];
	NSError *error = nil;
	XCTAssertTrue([server startWithError:&error], @"%@", error);
	[self addTeardownBlock:^{
		[server stop];
	}];
	id<NFKInferenceBackend> client = [NFKRemoteProvider backendForProvider:[NFKRemoteProvider.inferKit providerWithBaseURL:server.localBaseURL]
																	apiKey:nil modelName:@"weather-agent"];
	NFKInferenceResult *result = [client runInferenceForRequest:[self prompt:@"Weather?"] error:&error];
	XCTAssertEqualObjects(result.text, @"Clear.", @"%@", error);
	NFKToolInvocation *invocation = [[result outputForKey:NFKOutputToolInvocations] firstObject];
	XCTAssertTrue([invocation isKindOfClass:NFKToolInvocation.class], @"%@", [result outputForKey:NFKOutputToolInvocations]);
	XCTAssertEqualObjects(invocation.toolName, @"get_weather");
	XCTAssertEqualObjects(invocation.arguments, @{ @"city": @"Paris" });
	XCTAssertEqualObjects(invocation.result.text, @"18 C and clear in Paris");
	XCTAssertEqualObjects(invocation.result.outputs[@"celsius"], @18);
	XCTAssertEqual([[result outputForKey:NFKOutputMessages] count], 4u);
}

@end
