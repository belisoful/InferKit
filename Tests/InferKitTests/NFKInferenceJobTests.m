//
//  NFKInferenceJobTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceResult.h>

@interface NFKInferenceJobTests : XCTestCase
@end

@implementation NFKInferenceJobTests

- (void)testANewJobIsQueued
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	XCTAssertEqual(job.status, NFKInferenceJobStatusQueued);
	XCTAssertLessThan(job.progress, 0.0, @"progress is indeterminate until reported");
	XCTAssertNil(job.result);
	XCTAssertNil(job.error);
}

- (void)testReportingProgressMovesToRunning
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	[job reportProgress:0.5];
	XCTAssertEqual(job.status, NFKInferenceJobStatusRunning);
	XCTAssertEqual(job.progress, 0.5);
}

- (void)testReportingPartialResultsStreamsAndPersists
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	__block NSUInteger updates = 0;
	job.progressHandler = ^(NFKInferenceJob *j) { updates += 1; };

	[job reportProgress:0.3 partialResult:[NFKInferenceResult resultWithOutputs:@{ @"text": @"He" }]];
	XCTAssertEqualObjects([job.partialResult outputForKey:@"text"], @"He");

	[job reportProgress:0.6 partialResult:[NFKInferenceResult resultWithOutputs:@{ @"text": @"Hello" }]];
	XCTAssertEqualObjects([job.partialResult outputForKey:@"text"], @"Hello");

	// A plain progress report keeps the last partial rather than clearing it.
	[job reportProgress:0.7];
	XCTAssertEqualObjects([job.partialResult outputForKey:@"text"], @"Hello");
	XCTAssertEqual(updates, (NSUInteger)3);
}

- (void)testFinishingWithAResultSucceedsAndFiresCompletion
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	__block NFKInferenceJob *completed = nil;
	job.completionHandler = ^(NFKInferenceJob *j) { completed = j; };

	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{ @"image": @"out" }];
	[job finishWithResult:result];

	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded);
	XCTAssertEqualObjects(job.result, result);
	XCTAssertEqual(job.progress, 1.0);
	XCTAssertEqual(completed, job);
}

- (void)testFinishingWithAnErrorFails
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	NSError *error = [NSError errorWithDomain:@"x" code:1 userInfo:nil];
	[job finishWithError:error];
	XCTAssertEqual(job.status, NFKInferenceJobStatusFailed);
	XCTAssertEqualObjects(job.error, error);
}

- (void)testCancelRunsTheCancellationHandlerAndCompletion
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	__block BOOL cancelled = NO;
	__block BOOL completed = NO;
	job.cancellationHandler = ^{ cancelled = YES; };
	job.completionHandler = ^(NFKInferenceJob *j) { completed = YES; };

	[job cancel];

	XCTAssertEqual(job.status, NFKInferenceJobStatusCancelled);
	XCTAssertTrue(cancelled);
	XCTAssertTrue(completed);
}

- (void)testTerminalStateIsFinal
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	[job finishWithResult:[NFKInferenceResult resultWithOutputs:@{}]];
	[job finishWithError:[NSError errorWithDomain:@"x" code:1 userInfo:nil]];
	[job cancel];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"later producer calls are ignored");
}

- (void)testCompletionSetAfterFinishFiresImmediately
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	[job finishWithResult:[NFKInferenceResult resultWithOutputs:@{}]];

	__block BOOL fired = NO;
	job.completionHandler = ^(NFKInferenceJob *j) { fired = YES; };
	XCTAssertTrue(fired, @"a completion set on an already-finished job fires at once");
}

@end
