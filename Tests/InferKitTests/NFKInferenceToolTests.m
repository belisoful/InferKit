//
//  NFKInferenceToolTests.m
//  InferKitTests
//
//  A tool's declaration, the argument check that runs before its handler, the error results a
//  failing call becomes, and the caller each way of running a tool passes.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKInferenceTool.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

/*! A subclass that answers with its caller's client name. */
@interface NFKCallerNamingTool : NFKInferenceTool
@end

@implementation NFKCallerNamingTool

- (nullable NFKToolResult *)resultForArguments:(NSDictionary<NSString *, id> *)arguments
										caller:(nullable NFKToolCaller *)caller
										 error:(NSError **)error
{
	return [NFKToolResult resultWithText:caller.clientName ?: @"no caller"];
}

@end

@interface NFKInferenceToolTests : XCTestCase
@property (nonatomic, copy) NSDictionary *schema;
@end

@implementation NFKInferenceToolTests

- (void)setUp
{
	[super setUp];
	self.schema = @{ @"type": @"object",
					 @"properties": @{ @"city": @{ @"type": @"string" },
									   @"days": @{ @"type": @"integer" },
									   @"metric": @{ @"type": @"boolean" },
									   @"scale": @{ @"type": @"number" } },
					 @"required": @[ @"city" ] };
}

- (NFKInferenceTool *)weatherToolCounting:(NSInteger *)calls
{
	return [NFKInferenceTool toolWithName:@"get_weather" description:@"Current weather in a city." parameters:self.schema
								  handler:^NFKToolResult *(NSDictionary *arguments, NSError **error) {
		*calls += 1;
		return [NFKToolResult resultWithText:[NSString stringWithFormat:@"Sunny in %@", arguments[@"city"]]];
	}];
}

- (void)testTheDeclarationIsTheToolsEntry
{
	NSInteger calls = 0;
	NFKInferenceTool *tool = [self weatherToolCounting:&calls];
	NSDictionary *expected = @{ @"name": @"get_weather", @"description": @"Current weather in a city.", @"parameters": self.schema };
	XCTAssertEqualObjects(tool.declaration, expected);
}

- (void)testValidArgumentsRunTheHandler
{
	NSInteger calls = 0;
	NSDictionary *arguments = @{ @"city": @"Paris", @"days": @3, @"metric": @YES, @"scale": @1.5 };
	NFKToolResult *result = [[self weatherToolCounting:&calls] invokeWithArguments:arguments];
	XCTAssertEqual(calls, 1);
	XCTAssertFalse(result.isError);
	XCTAssertEqualObjects(result.text, @"Sunny in Paris");
	XCTAssertEqualObjects(result.outputs, @{});
	XCTAssertEqualObjects(result.imagePNGs, @[]);
}

- (void)testAMissingRequiredArgumentIsReportedWithoutRunning
{
	NSInteger calls = 0;
	NFKToolResult *result = [[self weatherToolCounting:&calls] invokeWithArguments:@{ @"days": @3 }];
	XCTAssertEqual(calls, 0);
	XCTAssertTrue(result.isError);
	XCTAssertTrue([result.text containsString:@"\"city\" is missing"], @"%@", result.text);
}

// A JSON boolean decodes as an NSNumber, so the check reads the number's own type to tell a boolean
// from a number.
- (void)testEachTopLevelTypeIsChecked
{
	NSInteger calls = 0;
	NFKInferenceTool *tool = [self weatherToolCounting:&calls];
	NSArray<NSString *> *problems = [tool problemsWithArguments:@{ @"city": @7, @"days": @2.5, @"metric": @1, @"scale": @YES }];
	XCTAssertEqual(problems.count, 4u, @"%@", problems);
	NSDictionary *integral = @{ @"city": @"Oslo", @"days": @2.0, @"metric": @NO, @"scale": @3 };
	XCTAssertEqualObjects([tool problemsWithArguments:integral], @[], @"an integral double is an integer and an integer is a number");
	NSDictionary *unnamed = @{ @"city": @"Oslo", @"unknown": @[ @1 ] };
	XCTAssertEqualObjects([tool problemsWithArguments:unnamed], @[], @"an argument the schema does not name is the tool's to judge");
	NSDictionary *nulled = @{ @"city": @"Oslo", @"days": NSNull.null };
	XCTAssertEqualObjects([tool problemsWithArguments:nulled], @[], @"null is how a model leaves an optional argument out");
}

- (void)testAHandlerErrorBecomesAnErrorResult
{
	NFKInferenceTool *tool = [NFKInferenceTool toolWithName:@"fails" description:@"d" parameters:@{ @"type": @"object" }
													handler:^NFKToolResult *(NSDictionary *arguments, NSError **error) {
		*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceBackendFailure
								 userInfo:@{ NSLocalizedDescriptionKey: @"the disk is full" }];
		return nil;
	}];
	NFKToolResult *result = [tool invokeWithArguments:@{}];
	XCTAssertTrue(result.isError);
	XCTAssertEqualObjects(result.text, @"fails failed: the disk is full");
}

- (void)testASubclassRunsItsOwnOverride
{
	NFKInferenceTool *tool = [[NFKInferenceTool alloc] initWithName:@"bare" description:@"d" parameters:@{ @"type": @"object" }];
	NFKToolResult *result = [tool invokeWithArguments:@{}];
	XCTAssertTrue(result.isError, @"a tool built without a handler reports that it has nothing to run");
}

- (void)testAResultCarriesTheApplicationsOutputsAndTheModelsImages
{
	NSData *png = [NSData dataWithBytes:"\x89PNG" length:4];
	NFKToolResult *result = [NFKToolResult resultWithText:@"made it" outputs:@{ NFKOutputText: @"for the app" } imagePNGs:@[ png ]];
	XCTAssertEqualObjects(result.outputs[NFKOutputText], @"for the app");
	XCTAssertEqualObjects(result.imagePNGs, @[ png ]);
	XCTAssertTrue([NFKToolResult errorResultWithText:@"no"].isError);
}

- (NFKToolCaller *)callerNamed:(NSString *)name
{
	return [[NFKToolCaller alloc] initWithSessionIdentifier:@"S1" clientName:name clientVersion:@"1.0" keyName:@"desk"
											  clientAddress:@"127.0.0.1" headers:@{ @"X-Harness-Name": @"left" }];
}

- (void)testACallerReadsItsHeadersWithoutRegardToCase
{
	NFKToolCaller *caller = [self callerNamed:@"claude-code"];
	XCTAssertEqualObjects(caller.headers, @{ @"x-harness-name": @"left" });
	XCTAssertEqualObjects([caller valueForHeader:@"X-HARNESS-NAME"], @"left");
	XCTAssertNil([caller valueForHeader:@"authorization"]);
}

- (void)testACallerHandlerReceivesTheCallerAndNilWhenInvokedDirectly
{
	NSMutableArray *callers = [NSMutableArray array];
	NFKInferenceTool *tool = [NFKInferenceTool toolWithName:@"who" description:@"d" parameters:self.schema
											  callerHandler:^NFKToolResult *(NSDictionary *arguments, NFKToolCaller *caller, NSError **error) {
		[callers addObject:caller ?: NSNull.null];
		return [NFKToolResult resultWithText:caller.clientName ?: @"no caller"];
	}];
	NFKToolCaller *caller = [self callerNamed:@"codex"];
	XCTAssertEqualObjects([tool invokeWithArguments:@{ @"city": @"Oslo" } caller:caller].text, @"codex");
	XCTAssertEqualObjects([tool invokeWithArguments:@{ @"city": @"Oslo" }].text, @"no caller");
	XCTAssertEqualObjects([tool resultForArguments:@{ @"city": @"Oslo" } error:NULL].text, @"no caller",
						  @"the method without a caller runs the caller handler too");
	XCTAssertTrue([tool invokeWithArguments:@{} caller:caller].isError, @"the schema check runs before the handler");
	XCTAssertEqualObjects(callers, (@[ caller, NSNull.null, NSNull.null ]));
}

- (void)testASubclassReadsItsCallerAndAnOlderOverrideStillRuns
{
	NFKCallerNamingTool *naming = [[NFKCallerNamingTool alloc] initWithName:@"naming" description:@"d" parameters:@{ @"type": @"object" }];
	XCTAssertEqualObjects([naming invokeWithArguments:@{} caller:[self callerNamed:@"gemini-cli"]].text, @"gemini-cli");
	XCTAssertEqualObjects([naming invokeWithArguments:@{}].text, @"no caller");

	NSInteger calls = 0;
	NFKInferenceTool *weather = [self weatherToolCounting:&calls];
	XCTAssertEqualObjects([weather invokeWithArguments:@{ @"city": @"Lima" } caller:[self callerNamed:@"x"]].text, @"Sunny in Lima");
	__block NFKToolResult *started = nil;
	XCTAssertNil([weather startWithArguments:@{ @"city": @"Rome" } caller:nil completion:^(NFKToolResult *result) {
		started = result;
	}], @"a tool without a start handler has nothing to cancel");
	XCTAssertEqualObjects(started.text, @"Sunny in Rome", @"and completes before returning");
	XCTAssertEqual(calls, 2);
}

- (void)testAStartHandlerStartsWithoutWaitingAndInvokingWaitsForIt
{
	__block NSUInteger starts = 0;
	__block BOOL cancelled = NO;
	NFKInferenceTool *tool = [NFKInferenceTool toolWithName:@"later" description:@"d" parameters:self.schema
											   startHandler:^dispatch_block_t (NSDictionary *arguments, NFKToolCaller *caller,
																			   void (^completion)(NFKToolResult *)) {
		starts += 1;
		NSString *text = [NSString stringWithFormat:@"%@ for %@", arguments[@"city"], caller.clientName ?: @"no caller"];
		dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.05 * NSEC_PER_SEC)), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			completion([NFKToolResult resultWithText:text]);
		});
		return ^{
			cancelled = YES;
		};
	}];
	XCTestExpectation *finished = [self expectationWithDescription:@"the call completes later"];
	__block NFKToolResult *result = nil;
	dispatch_block_t cancel = [tool startWithArguments:@{ @"city": @"Kyiv" } caller:[self callerNamed:@"opencode"] completion:^(NFKToolResult *finalResult) {
		result = finalResult;
		[finished fulfill];
	}];
	XCTAssertNil(result, @"the start returns before the result");
	XCTAssertNotNil(cancel);
	[self waitForExpectations:@[ finished ] timeout:10];
	XCTAssertEqualObjects(result.text, @"Kyiv for opencode");
	cancel();
	XCTAssertTrue(cancelled);

	XCTAssertEqualObjects([tool invokeWithArguments:@{ @"city": @"Riga" }].text, @"Riga for no caller", @"invoking waits for the completion");
	__block NFKToolResult *refused = nil;
	XCTAssertNil([tool startWithArguments:@{} caller:nil completion:^(NFKToolResult *finalResult) {
		refused = finalResult;
	}]);
	XCTAssertTrue(refused.isError, @"arguments that fail the schema check complete at once without a start");
	XCTAssertEqual(starts, 2u);
}

@end
