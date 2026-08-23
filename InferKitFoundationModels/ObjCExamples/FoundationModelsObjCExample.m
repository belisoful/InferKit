//
//  FoundationModelsObjCExample.m
//  InferKitFoundationModelsObjCExamples
//
//  The Objective-C half of this package's examples, mirroring the "Foundation Models" section of
//  Docs/examples.md. The backend, its tools, and its typed parameters are all `@objc`, so an
//  Objective-C app (an FCPX plugin, an AppKit app) drives Apple's on-device model through the same
//  NFKInferenceBackend protocol it uses for every other engine, without writing Swift.
//
//  Generation needs Apple Intelligence enabled, so these exercise construction and the contract; the
//  Swift half covers the same ground and skips its generating tests the same way.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
@import InferKitFoundationModels;

@interface FoundationModelsObjCExample : XCTestCase
@end

@implementation FoundationModelsObjCExample

- (void)testObjectiveCBuildsTheBackendAndReadsItsContract
{
	NFKFoundationModelsBackend *backend = [[NFKFoundationModelsBackend alloc] init];
	XCTAssertEqualObjects(backend.backendIdentifier, @"foundation-models");
	// `isReady` mirrors SystemLanguageModel.default.availability, so it is false wherever Apple
	// Intelligence is off — which is the check to make before offering the feature in a UI.
	(void)backend.isReady;
}

- (void)testObjectiveCRegistersATypedToolTheModelCanCall
{
	// A tool is a name, a description, typed parameters, and a handler. The adapter builds the
	// runtime schema, so no compile-time @Generable type is needed.
	NFKFoundationToolParameter *city =
		[[NFKFoundationToolParameter alloc] initWithName:@"city"
											 description:@"The city to report on"
													type:NFKToolParameterTypeString
												required:YES];

	// Objective-C gets the synchronous initializer; the asynchronous handler is Swift-only, because a
	// block cannot carry Swift's `async throws`.
	NFKFoundationTool *weather =
		[[NFKFoundationTool alloc] initWithName:@"lookup_weather"
									description:@"Look up the current weather for a city"
									 parameters:@[city]
									syncHandler:^NSString * _Nonnull(NSDictionary<NSString *, id> * _Nonnull arguments) {
											return [NSString stringWithFormat:@"It is fair in %@.", arguments[@"city"]];
										}];

	NFKFoundationModelsBackend *backend = [[NFKFoundationModelsBackend alloc] init];
	backend.tools = @[weather];
	XCTAssertEqual(backend.tools.count, (NSUInteger)1);
	XCTAssertEqualObjects(backend.tools.firstObject.name, @"lookup_weather");

	XCTAssertEqualObjects(weather.parameters.firstObject.name, @"city");
	XCTAssertEqual(weather.parameters.firstObject.type, NFKToolParameterTypeString);
	XCTAssertTrue(weather.parameters.firstObject.isRequired);
}

- (void)testObjectiveCAsksForStructuredOutput
{
	// Setting a response schema switches generation to the structured path: the result carries the
	// parsed dictionary under the core key NFKOutputStructured and the JSON under NFKOutputText.
	NFKFoundationToolParameter *title =
		[[NFKFoundationToolParameter alloc] initWithName:@"title"
											 description:@"A short headline"
													type:NFKToolParameterTypeString
												required:YES];
	NFKFoundationToolParameter *rating =
		[[NFKFoundationToolParameter alloc] initWithName:@"rating"
											 description:@"A score from 1 to 5"
													type:NFKToolParameterTypeInteger
												required:NO];

	NFKFoundationModelsBackend *backend = [[NFKFoundationModelsBackend alloc] init];
	backend.responseSchema = @[title, rating];
	XCTAssertEqual(backend.responseSchema.count, (NSUInteger)2);

	// The request is the same shape as for any other backend.
	NFKInferenceRequest *request =
		[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Review this lens in one line." }];
	XCTAssertEqualObjects(request.prompt, @"Review this lens in one line.");
}

- (void)testObjectiveCReachesTheBackendThroughDynamicDiscovery
{
	// Linking this package ships NFKFoundationModelsProvider under the name the core tries for its
	// text-generation capability, so the core activates it with no registration call.
	XCTAssertTrue([NFKDynamicBackend isCapabilityAvailable:NFKCapabilityTextGeneration]);

	NSError *error = nil;
	id<NFKInferenceBackend> backend = [NFKDynamicBackend backendForCapability:NFKCapabilityTextGeneration
																		error:&error];
	XCTAssertNotNil(backend, @"%@", error);
	XCTAssertEqualObjects(backend.backendIdentifier, @"foundation-models");
}

@end
