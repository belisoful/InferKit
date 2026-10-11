//
//  NFKInferenceToolTests.m
//  InferKitTests
//
//  A tool's declaration, the argument check that runs before its handler, and the error results a
//  failing call becomes.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKInferenceTool.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

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

@end
