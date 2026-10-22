//
//  NFKModelSanityCheckTests.m
//  InferKitTests
//
//  The post-load sanity check: the verdicts on fluent, empty, garbled, and looping text, a number that is
//  no loop, and one greedy generation run through a backend that answers a fixed reply.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

/*! A backend that answers every request with its reply, and keeps the request it was last given. */
@interface NFKSanityReplyBackend : NSObject <NFKInferenceBackend>
@property (nonatomic, copy, nullable) NSString *reply;
@property (nonatomic, strong, nullable) NFKInferenceRequest *lastRequest;
@end

@implementation NFKSanityReplyBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"sanity-reply";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	self.lastRequest = request;
	return [NFKInferenceResult resultWithOutputs:self.reply != nil ? @{ NFKOutputText: self.reply } : @{}];
}

@end

@interface NFKModelSanityCheckTests : XCTestCase
@end

@implementation NFKModelSanityCheckTests

- (void)testAFluentReplyPasses
{
	for (NSString *reply in @[ @" Paris. It is the largest city in France and its seat of government.",
							   @" Paris, home to 2100000 people in the city proper." ]) {
		NFKModelSanityResult *result = [NFKModelSanityCheck resultForOutput:reply];
		XCTAssertTrue(result.isPassed, @"%@", result);
		XCTAssertEqual(result.findings, NFKModelSanityFindingNone);
	}
}

- (void)testAnEmptyReplyIsFound
{
	XCTAssertEqual([NFKModelSanityCheck resultForOutput:@"  \n "].findings, NFKModelSanityFindingEmpty);
}

- (void)testUndecodedBytesAreFound
{
	XCTAssertEqual([NFKModelSanityCheck resultForOutput:@"Par�is"].findings, NFKModelSanityFindingGarbled);
	XCTAssertEqual([NFKModelSanityCheck resultForOutput:@"Paris\a"].findings, NFKModelSanityFindingGarbled);
}

- (void)testALoopIsFound
{
	NSArray<NSString *> *loops = @[ @"the the the the the the the the",
									@"I am a model. I am a model. I am a model. I am a model.",
									@"!!!!!!!!!!!!",
									@"ababababababababababababababababab" ];
	for (NSString *loop in loops) {
		XCTAssertEqual([NFKModelSanityCheck resultForOutput:loop].findings, NFKModelSanityFindingRepetitive, @"%@", loop);
	}
}

- (void)testACheckRunsOneGreedyGeneration
{
	NFKSanityReplyBackend *backend = [[NFKSanityReplyBackend alloc] init];
	backend.reply = @" Paris.";
	NSError *error = nil;
	NFKModelSanityResult *result = [NFKModelSanityCheck checkBackend:backend error:&error];
	XCTAssertTrue(result.isPassed, @"%@ %@", result, error);
	XCTAssertEqualObjects([backend.lastRequest inputForKey:NFKInputPrompt], NFKModelSanityCheck.defaultPrompt);
	XCTAssertEqualObjects(backend.lastRequest.parameters[NFKParameterTemperature], @0);
	XCTAssertEqualObjects(backend.lastRequest.parameters[NFKParameterMaxTokens], @(NFKModelSanityCheck.defaultTokenCount));
}

- (void)testAResultWithoutTextFailsTheCheck
{
	NSError *error = nil;
	XCTAssertNil([NFKModelSanityCheck checkBackend:[[NFKSanityReplyBackend alloc] init] error:&error]);
	XCTAssertEqualObjects(error.domain, NFKInferenceErrorDomain);
}

@end
