//
//  NFKDynamicBackendTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

#pragma mark Test providers

// A backend the test provider hands back — stands in for a consumer's Stable Diffusion wrapper.
@interface NFKTestDynamicBackend : NSObject <NFKInferenceBackend>
@end

@implementation NFKTestDynamicBackend
- (BOOL)isReady { return YES; }
- (NSString *)backendIdentifier { return @"test-dynamic"; }
- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												 error:(NSError **)error
{
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"ok" }];
}
- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	[job finishWithResult:[NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"ok" }]];
	return job;
}
@end

// A conforming provider that InferKit discovers by name.
@interface NFKTestProvider : NSObject <NFKDynamicBackendProvider>
@end

@implementation NFKTestProvider
+ (nullable id<NFKInferenceBackend>)makeInferenceBackend { return [[NFKTestDynamicBackend alloc] init]; }
@end

// A class that does NOT conform to the provider protocol.
@interface NFKNonProvider : NSObject
@end
@implementation NFKNonProvider
@end

#pragma mark Tests

@interface NFKDynamicBackendTests : XCTestCase
@end

@implementation NFKDynamicBackendTests

- (void)testAPresentProviderIsDiscoveredAndBuildsABackend
{
	XCTAssertTrue([NFKDynamicBackend isProviderAvailable:@"NFKTestProvider"]);
	NSError *error = nil;
	id<NFKInferenceBackend> backend = [NFKDynamicBackend backendForProviderClassName:@"NFKTestProvider" error:&error];
	XCTAssertNotNil(backend);
	XCTAssertNil(error);
	XCTAssertEqualObjects(backend.backendIdentifier, @"test-dynamic");
}

- (void)testAnAbsentClassResolvesToNilWithoutCrashing
{
	XCTAssertFalse([NFKDynamicBackend isProviderAvailable:@"NFKClassThatIsNotLinked"]);
	NSError *error = nil;
	id<NFKInferenceBackend> backend = [NFKDynamicBackend backendForProviderClassName:@"NFKClassThatIsNotLinked" error:&error];
	XCTAssertNil(backend, @"an unlinked engine is simply unavailable");
	XCTAssertEqualObjects(error.domain, NFKInferenceErrorDomain);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
}

- (void)testANonConformingClassIsNotTreatedAsAProvider
{
	XCTAssertFalse([NFKDynamicBackend isProviderAvailable:@"NFKNonProvider"],
				   @"a class must conform to NFKDynamicBackendProvider to be a provider");
}

- (void)testCapabilityResolvesARegisteredProvider
{
	[NFKDynamicBackend registerProviderClassName:@"NFKTestProvider" forCapability:@"unit-test-capability"];
	XCTAssertTrue([NFKDynamicBackend isCapabilityAvailable:@"unit-test-capability"]);
	NSError *error = nil;
	id<NFKInferenceBackend> backend = [NFKDynamicBackend backendForCapability:@"unit-test-capability" error:&error];
	XCTAssertNotNil(backend);
	XCTAssertNil(error);
}

// Registration is process-wide, so the unavailable-then-registered transition is checked in one
// deterministic test rather than split across two order-dependent ones.
- (void)testStableDiffusionActivatesOnlyOnceAProviderIsLinked
{
	// Nothing is registered and no NFKStableDiffusionProvider class is linked, so it starts unavailable.
	XCTAssertFalse([NFKDynamicBackend isCapabilityAvailable:NFKCapabilityStableDiffusion]);
	NSError *absentError = nil;
	XCTAssertNil([NFKDynamicBackend stableDiffusionBackendWithError:&absentError]);
	XCTAssertEqual(absentError.code, kNFKError_InferenceUnsupported);

	// Registering a present provider under the built-in capability activates it.
	[NFKDynamicBackend registerProviderClassName:@"NFKTestProvider" forCapability:NFKCapabilityStableDiffusion];
	XCTAssertTrue([NFKDynamicBackend isCapabilityAvailable:NFKCapabilityStableDiffusion]);
	NSError *presentError = nil;
	id<NFKInferenceBackend> backend = [NFKDynamicBackend stableDiffusionBackendWithError:&presentError];
	XCTAssertNotNil(backend, @"a registered, present provider activates the Stable Diffusion capability");
	XCTAssertNil(presentError);
	XCTAssertEqualObjects(backend.backendIdentifier, @"test-dynamic");
}

@end
