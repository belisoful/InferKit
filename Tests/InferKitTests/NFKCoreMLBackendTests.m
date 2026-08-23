//
//  NFKCoreMLBackendTests.m
//  NFKTests
//
//  Covers the backend's contract and error paths. A live model run is host-verified.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKCoreMLBackend.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKErrors.h>

@interface NFKCoreMLBackendTests : XCTestCase
@end

@implementation NFKCoreMLBackendTests

- (void)testTheBackendReportsItsIdentifier
{
	NFKCoreMLBackend *backend = [NFKCoreMLBackend backendWithModelURL:nil];
	XCTAssertEqualObjects(backend.backendIdentifier, @"coreml");
}

- (void)testANewBackendIsNotReady
{
	NFKCoreMLBackend *backend = [NFKCoreMLBackend backendWithModelURL:nil];
	XCTAssertFalse(backend.isReady);
}

- (void)testBackendWithModelURLStoresTheURL
{
	NSURL *url = [NSURL fileURLWithPath:@"/models/example.mlmodelc"];
	NFKCoreMLBackend *backend = [NFKCoreMLBackend backendWithModelURL:url];
	XCTAssertEqualObjects(backend.modelURL, url);
}

- (void)testPreparingWithoutAURLFails
{
	NFKCoreMLBackend *backend = [NFKCoreMLBackend backendWithModelURL:nil];
	NSError *error = nil;
	XCTAssertFalse([backend prepareWithError:&error]);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceNotReady);
}

- (void)testRunningWithoutAModelFails
{
	NFKCoreMLBackend *backend = [NFKCoreMLBackend backendWithModelURL:nil];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{}];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
	XCTAssertNil(result);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceNotReady);
}

- (void)testLoadingAMissingModelFails
{
	NFKCoreMLBackend *backend = [NFKCoreMLBackend backendWithModelURL:nil];
	NSURL *missing = [NSURL fileURLWithPath:@"/nonexistent/does-not-exist.mlmodelc"];
	NSError *error = nil;
	XCTAssertFalse([backend loadModelFromURL:missing error:&error]);
	XCTAssertNotNil(error);
	XCTAssertFalse(backend.isReady);
}

@end
