//
//  NFKInferenceCoreTests.m
//  NFKTests
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceBackend.h>
#import <InferKit/NFKPassthroughBackend.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKCoreMLLanguageBackend.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

@interface NFKInferenceCoreTests : XCTestCase
@end

@implementation NFKInferenceCoreTests

#pragma mark Request

- (void)testARequestKeepsInputsAndParametersSeparate
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ @"image": @"plate" }
																	 parameters:@{ @"seed": @42 }];
	XCTAssertEqualObjects([request inputForKey:@"image"], @"plate");
	XCTAssertEqualObjects([request parameterForKey:@"seed"], @42);
	XCTAssertNil([request inputForKey:@"seed"], @"a parameter is not an input");
	XCTAssertNil([request parameterForKey:@"image"], @"an input is not a parameter");
}

- (void)testARequestDefaultsToEmptyCollections
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{}];
	XCTAssertEqualObjects(request.inputs, @{});
	XCTAssertEqualObjects(request.parameters, @{});
}

- (void)testRequestEqualityConsidersInputsAndParameters
{
	NFKInferenceRequest *a = [NFKInferenceRequest requestWithInputs:@{ @"image": @"plate" } parameters:@{ @"seed": @1 }];
	NFKInferenceRequest *b = [NFKInferenceRequest requestWithInputs:@{ @"image": @"plate" } parameters:@{ @"seed": @1 }];
	NFKInferenceRequest *c = [NFKInferenceRequest requestWithInputs:@{ @"image": @"plate" } parameters:@{ @"seed": @2 }];
	XCTAssertEqualObjects(a, b);
	XCTAssertNotEqualObjects(a, c);
}

- (void)testARequestIsImmutableUnderCopy
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ @"image": @"plate" }];
	XCTAssertEqual([request copy], request, @"an immutable value copies to itself");
}

#pragma mark Result

- (void)testAResultExposesItsOutputs
{
	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{ @"image": @"generated" }];
	XCTAssertEqualObjects([result outputForKey:@"image"], @"generated");
	XCTAssertNil([result outputForKey:@"missing"]);
}

#pragma mark Passthrough backend

- (void)testThePassthroughBackendIsAlwaysReady
{
	NFKPassthroughBackend *backend = [NFKPassthroughBackend backend];
	XCTAssertTrue(backend.isReady);
	XCTAssertEqualObjects(backend.backendIdentifier, @"passthrough");
}

- (void)testThePassthroughBackendEchoesInputsByDefault
{
	NFKPassthroughBackend *backend = [NFKPassthroughBackend backend];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ @"image": @"plate" }];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
	XCTAssertNil(error);
	XCTAssertEqualObjects([result outputForKey:@"image"], @"plate");
}

- (void)testThePassthroughBackendRoutesThroughItsOutputMap
{
	NFKPassthroughBackend *backend = [NFKPassthroughBackend backend];
	backend.outputMap = @{ @"fg": @"rgb", @"matte": @"hint" };
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ @"rgb": @"plate", @"hint": @"alpha" }];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
	XCTAssertNil(error);
	XCTAssertEqualObjects([result outputForKey:@"fg"], @"plate");
	XCTAssertEqualObjects([result outputForKey:@"matte"], @"alpha");
	XCTAssertNil([result outputForKey:@"rgb"], @"only mapped outputs are produced");
}

- (void)testThePassthroughBackendFailsOnAMappedInputThatIsMissing
{
	NFKPassthroughBackend *backend = [NFKPassthroughBackend backend];
	backend.outputMap = @{ @"fg": @"rgb" };
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ @"hint": @"alpha" }];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
	XCTAssertNil(result);
	XCTAssertNotNil(error);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceMissingInput);
}

- (void)testThePassthroughBackendConformsToTheBackendProtocol
{
	id<NFKInferenceBackend> backend = [NFKPassthroughBackend backend];
	XCTAssertTrue([backend conformsToProtocol:@protocol(NFKInferenceBackend)]);
	XCTAssertTrue([backend respondsToSelector:@selector(runInferenceForRequest:error:)]);
}

#pragma mark Declared keys

- (void)testTheRemoteBackendDeclaresTheKeysItActsOn
{
	id<NFKInferenceBackend> backend = [NFKRemoteBackend backendWithEndpointURL:nil];
	XCTAssertTrue([backend.supportedParameterKeys containsObject:NFKParameterTools]);
	XCTAssertTrue([backend.supportedParameterKeys containsObject:NFKParameterJSONSchema]);
	XCTAssertTrue([backend.supportedInputKeys containsObject:NFKInputMessages]);
	XCTAssertTrue([backend.supportedInputKeys containsObject:NFKInputImage], @"a vision model reads an image");
	XCTAssertTrue([backend.supportedParameterKeys containsObject:NFKParameterReasoningEffort]);
	// The text parameters are renamed to the endpoint's spelling, so the contract's keys reach it.
	for (NSString *key in @[ NFKParameterMaxTokens, NFKParameterTopP, NFKParameterTopK,
							 NFKParameterStopSequences, NFKParameterRepetitionPenalty ]) {
		XCTAssertTrue([backend.supportedParameterKeys containsObject:key], @"%@", key);
	}
}

- (void)testTheCoreMLLanguageBackendDeclaresNoSchemaKey
{
	if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
		id<NFKInferenceBackend> backend = [NFKCoreMLLanguageBackend backendWithModelDirectoryURL:nil];
		XCTAssertTrue([backend.supportedParameterKeys containsObject:NFKParameterChoices]);
		XCTAssertTrue([backend.supportedParameterKeys containsObject:NFKParameterTopK]);
		XCTAssertFalse([backend.supportedParameterKeys containsObject:NFKParameterJSONSchema],
					   @"the token grammar is reached through the output format and the choices");
		XCTAssertFalse([backend.supportedInputKeys containsObject:NFKInputImage]);
	}
}

- (void)testABackendNeedNotDeclareItsKeys
{
	id<NFKInferenceBackend> backend = [NFKPassthroughBackend backend];
	XCTAssertFalse([backend respondsToSelector:@selector(supportedParameterKeys)]);
	XCTAssertFalse([backend respondsToSelector:@selector(supportedInputKeys)]);
}

#pragma mark Preparing through the protocol

- (void)testPreparingABackendWithNothingToLoadSucceeds
{
	NSError *error = nil;
	XCTAssertTrue(NFKInferencePrepare([NFKPassthroughBackend backend], &error));
	XCTAssertNil(error);
}

- (void)testPreparingReportsTheBackendsError
{
	if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
		NSError *error = nil;
		XCTAssertFalse(NFKInferencePrepare([NFKCoreMLLanguageBackend backendWithModelDirectoryURL:nil], &error));
		XCTAssertNotNil(error);
	}
}

@end
