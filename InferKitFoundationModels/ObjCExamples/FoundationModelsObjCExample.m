//
//  FoundationModelsObjCExample.m
//  InferKitFoundationModelsObjCExamples
//
//  The Objective-C half of this package's examples, mirroring the "Foundation Models" section of
//  Docs/examples.md. The backend and its tools are `@objc`, and every option is a core request key,
//  so an Objective-C app (an FCPX plugin, an AppKit app) drives Apple's on-device model through the
//  same NFKInferenceBackend protocol it uses for every other engine, without writing Swift.
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
	XCTAssertGreaterThan(backend.contextSize, (NSInteger)0);
}

- (void)testObjectiveCChoosesTheModel
{
	// The on-device model is the default; its specialization and guardrails are plain enums. Private
	// Cloud Compute is macOS 27 / iOS 27: below that the backend is not ready and a request fails
	// with kNFKError_InferenceUnsupported rather than running on the device unasked.
	NFKFoundationModelsBackend *backend = [[NFKFoundationModelsBackend alloc] init];
	XCTAssertEqual(backend.model, NFKFoundationModelOnDevice);
	backend.useCase = NFKFoundationModelUseCaseContentTagging;
	backend.guardrails = NFKFoundationModelGuardrailsPermissiveContentTransformations;
	XCTAssertEqual(backend.useCase, NFKFoundationModelUseCaseContentTagging);

	if (@available(macOS 27, iOS 27, *)) {
		// The quota is readable whatever `model` is set to, so an app decides before switching.
		NFKFoundationModelQuota *quota = backend.privateCloudComputeQuota;
		if (!quota.isLimitReached) {
			backend.model = NFKFoundationModelPrivateCloudCompute;
		}
	} else {
		backend.model = NFKFoundationModelPrivateCloudCompute;
		XCTAssertFalse(backend.isReady);
		NSError *error = nil;
		XCTAssertFalse([backend prepareWithError:&error]);
		XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
	}
}

- (void)testObjectiveCRegistersAToolTheModelCanCall
{
	// A tool is a name, a description, a JSON Schema for its arguments, and a handler. The schema
	// is the same dictionary a remote backend takes in an NFKParameterTools entry.
	NSDictionary *parameters = @{
		@"type": @"object",
		@"properties": @{ @"city": @{ @"type": @"string", @"description": @"The city to report on" } },
		@"required": @[ @"city" ],
	};

	// Objective-C gets the synchronous initializer; the asynchronous handler is Swift-only, because a
	// block cannot carry Swift's `async throws`.
	NFKFoundationTool *weather =
		[[NFKFoundationTool alloc] initWithName:@"lookup_weather"
									description:@"Look up the current weather for a city"
									 parameters:parameters
									syncHandler:^NSString * _Nonnull(NSDictionary<NSString *, id> * _Nonnull arguments) {
											return [NSString stringWithFormat:@"It is fair in %@.", arguments[@"city"]];
										}];

	NFKFoundationModelsBackend *backend = [[NFKFoundationModelsBackend alloc] init];
	backend.tools = @[weather];
	XCTAssertEqual(backend.tools.count, (NSUInteger)1);
	XCTAssertEqualObjects(backend.tools.firstObject.name, @"lookup_weather");
	XCTAssertEqualObjects(weather.parameters[@"required"], @[ @"city" ]);
	XCTAssertEqualObjects(weather.declaration[@"name"], @"lookup_weather");
}

- (void)testObjectiveCAsksForStructuredOutputThroughTheCoreKey
{
	// NFKParameterJSONSchema switches generation to the structured path: the result carries the
	// parsed object under NFKOutputStructured and the JSON under NFKOutputText. The same request
	// runs against NFKRemoteBackend and the MLX language backend.
	NSDictionary *schema = @{
		@"type": @"object",
		@"properties": @{
			@"title": @{ @"type": @"string", @"description": @"A short headline" },
			@"rating": @{ @"type": @"integer", @"description": @"A score from 1 to 5", @"minimum": @1, @"maximum": @5 },
		},
		@"required": @[ @"title" ],
	};
	NFKInferenceRequest *request =
		[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Review this lens in one line." }
									parameters:@{ NFKParameterJSONSchema: schema, NFKParameterTemperature: @0 }];
	XCTAssertEqualObjects(request.prompt, @"Review this lens in one line.");
	XCTAssertEqualObjects(request.parameters[NFKParameterJSONSchema], schema);
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
