//
//  NFKComputePlanTests.m
//  NFKTests
//
//  The compute plan's contract without a model, plus an opt-in run against a real compiled one.
//  Core ML publishes no placement information before macOS 14.4 / iOS 17.4, and a caller has to be
//  able to tell "the OS cannot answer" from "the model is not on the Neural Engine" — those two are
//  what most of this covers.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKComputePlan.h>
#import <InferKit/NFKCoreMLBackend.h>
#import <InferKit/NFKErrors.h>

@interface NFKComputePlanTests : XCTestCase
@end

@implementation NFKComputePlanTests

- (nullable NSURL *)compiledModelURL
{
	NSString *path = NSProcessInfo.processInfo.environment[@"INFERKIT_TEST_MLMODELC"];
	return path.length > 0 ? [NSURL fileURLWithPath:path] : nil;
}

#pragma mark Availability

- (void)testAvailabilityFollowsTheOperatingSystem
{
	if (@available(macOS 14.4, iOS 17.4, tvOS 17.4, *)) {
		XCTAssertTrue(NFKComputePlan.isAvailable);
	} else {
		XCTAssertFalse(NFKComputePlan.isAvailable);
	}
}

// An older system reports that it cannot answer rather than reporting zero operations, which would
// read as "nothing is on the Neural Engine".
- (void)testAnUnsupportedSystemReportsUnsupportedRatherThanAnEmptyPlan
{
	if (NFKComputePlan.isAvailable) {
		return;
	}
	NSError *error = nil;
	NFKComputePlan *plan = [NFKComputePlan planForCompiledModelAtURL:[NSURL fileURLWithPath:@"/tmp/none.mlmodelc"]
													   computeUnits:MLComputeUnitsAll
															  error:&error];
	XCTAssertNil(plan);
	XCTAssertEqualObjects(error.domain, NFKInferenceErrorDomain);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
}

#pragma mark Failure paths

- (void)testAMissingURLFailsWithAnError
{
	NSError *error = nil;
	XCTAssertNil([NFKComputePlan planForCompiledModelAtURL:(NSURL * _Nonnull)nil
											  computeUnits:MLComputeUnitsAll
													 error:&error]);
	XCTAssertNotNil(error);
}

- (void)testAModelThatIsNotThereFailsWithoutCrashing
{
	NSURL *absent = [NSURL fileURLWithPath:NSTemporaryDirectory()];
	absent = [absent URLByAppendingPathComponent:@"inferkit-absent-model.mlmodelc"];
	NSError *error = nil;
	XCTAssertNil([NFKComputePlan planForCompiledModelAtURL:absent
											  computeUnits:MLComputeUnitsAll
													 error:&error]);
	XCTAssertNotNil(error);
}

// The error argument is optional, and a caller that ignores it must not crash.
- (void)testFailingWithANullErrorPointerIsSafe
{
	XCTAssertNil([NFKComputePlan planForCompiledModelAtURL:(NSURL * _Nonnull)nil
											  computeUnits:MLComputeUnitsAll
													 error:NULL]);
}

#pragma mark The non-blocking path

// Core ML's own compute-plan API is asynchronous; the blocking variant is a semaphore around this
// one, so a caller that cannot block has a path that does no waiting.
- (void)testTheCompletionHandlerVariantReportsFailureWithoutBlocking
{
	XCTestExpectation *answered = [self expectationWithDescription:@"the handler ran"];
	NSURL *absent = [[NSURL fileURLWithPath:NSTemporaryDirectory()]
					 URLByAppendingPathComponent:@"inferkit-absent-async.mlmodelc"];
	[NFKComputePlan planForCompiledModelAtURL:absent
								 computeUnits:MLComputeUnitsAll
							completionHandler:^(NFKComputePlan *plan, NSError *error) {
		XCTAssertNil(plan);
		XCTAssertNotNil(error);
		[answered fulfill];
	}];
	[self waitForExpectationsWithTimeout:30 handler:nil];
}

// A missing URL cannot even be attempted, and the handler still runs rather than being dropped.
- (void)testTheCompletionHandlerRunsEvenWhenThePlanCannotBeAttempted
{
	XCTestExpectation *answered = [self expectationWithDescription:@"the handler ran"];
	[NFKComputePlan planForCompiledModelAtURL:(NSURL * _Nonnull)nil
								 computeUnits:MLComputeUnitsAll
							completionHandler:^(NFKComputePlan *plan, NSError *error) {
		XCTAssertNil(plan);
		XCTAssertEqualObjects(error.domain, NFKInferenceErrorDomain);
		[answered fulfill];
	}];
	[self waitForExpectationsWithTimeout:10 handler:nil];
}

#pragma mark The backend's compute units

// MLComputeUnitsCPUOnly is zero, so an unset property would move every model to the CPU and look
// like a Neural Engine that does not work.
- (void)testTheCoreMLBackendDefaultsToAllComputeUnits
{
	XCTAssertEqual([[NFKCoreMLBackend alloc] init].computeUnits, MLComputeUnitsAll);
	XCTAssertEqual([NFKCoreMLBackend backendWithModelURL:nil].computeUnits, MLComputeUnitsAll);
}

- (void)testTheCoreMLBackendKeepsTheComputeUnitsItIsGiven
{
	NFKCoreMLBackend *backend = [NFKCoreMLBackend backendWithModelURL:nil];
	backend.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
	XCTAssertEqual(backend.computeUnits, MLComputeUnitsCPUAndNeuralEngine);
}

#pragma mark Opt-in: a real compiled model

// Set INFERKIT_TEST_MLMODELC to an .mlmodelc directory to see where its operations actually land.
- (void)testARealModelReportsWhereItsOperationsLand
{
	NSURL *url = [self compiledModelURL];
	if (url == nil || !NFKComputePlan.isAvailable) {
		return;		// opt-in: set INFERKIT_TEST_MLMODELC to run this
	}

	NSError *error = nil;
	NFKComputePlan *plan = [NFKComputePlan planForCompiledModelAtURL:url
													   computeUnits:MLComputeUnitsAll
															  error:&error];
	XCTAssertNotNil(plan, @"the plan failed to load: %@", error);
	XCTAssertGreaterThan(plan.operationCount, 0, @"a real model has operations to place");
	XCTAssertEqual(plan.operationCount,
				   plan.neuralEngineOperationCount + plan.gpuOperationCount
				   + plan.cpuOperationCount + plan.unknownOperationCount,
				   @"every operation is counted exactly once");
	XCTAssertGreaterThanOrEqual(plan.neuralEngineFraction, 0.0);
	XCTAssertLessThanOrEqual(plan.neuralEngineFraction, 1.0);
	NSLog(@"%@ -> %@", url.lastPathComponent, plan.describedPlacement);
	if (plan.operatorNamesOffNeuralEngine.count > 0) {
		NSLog(@"off the Neural Engine: %@",
			  [plan.operatorNamesOffNeuralEngine componentsJoinedByString:@", "]);
	}
}

// The same model under CPU-only must report nothing on the Neural Engine, which is what shows the
// plan reflects the requested compute units rather than a fixed property of the model.
- (void)testCPUOnlyPlacesNothingOnTheNeuralEngine
{
	NSURL *url = [self compiledModelURL];
	if (url == nil || !NFKComputePlan.isAvailable) {
		return;		// opt-in
	}
	NSError *error = nil;
	NFKComputePlan *plan = [NFKComputePlan planForCompiledModelAtURL:url
													   computeUnits:MLComputeUnitsCPUOnly
															  error:&error];
	XCTAssertNotNil(plan, @"the plan failed to load: %@", error);
	XCTAssertEqual(plan.neuralEngineOperationCount, 0);
	XCTAssertFalse(plan.runsEntirelyOnNeuralEngine);
}

@end
