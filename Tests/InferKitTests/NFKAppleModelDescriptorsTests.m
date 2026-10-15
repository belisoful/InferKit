//
//  NFKAppleModelDescriptorsTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

@interface NFKAppleModelDescriptorsTests : XCTestCase
@end

@implementation NFKAppleModelDescriptorsTests

- (NSSet<NSString *> *)keysOfPorts:(NSArray<NFKModelPort *> *)ports
{
	return [NSSet setWithArray:[ports valueForKey:@"key"]];
}

- (NSSet<NSString *> *)declaredKeysOf:(id<NFKInferenceBackend>)backend selector:(SEL)selector
{
	if (![backend respondsToSelector:selector]) {
		return [NSSet set];
	}
	return [(id)backend valueForKey:NSStringFromSelector(selector)];
}

#pragma mark Descriptors

- (void)testEveryDescriptorStatesTheKeysOfTheBackendBuiltForIt
{
	NSArray<NFKModelDescriptor *> *descriptors = [NFKAppleModelDescriptors modelDescriptors];
	XCTAssertGreaterThan(descriptors.count, (NSUInteger)0);
	for (NFKModelDescriptor *descriptor in descriptors) {
		id<NFKInferenceBackend> backend = [NFKAppleModelDescriptors backendNamed:descriptor.registryName];
		XCTAssertNotNil(backend, @"%@", descriptor.identifier);
		XCTAssertEqualObjects([self keysOfPorts:descriptor.inputs],
							  [self declaredKeysOf:backend selector:@selector(supportedInputKeys)], @"%@", descriptor.identifier);
		XCTAssertEqualObjects([self keysOfPorts:descriptor.outputs],
							  [self declaredKeysOf:backend selector:@selector(supportedOutputKeys)], @"%@", descriptor.identifier);
		XCTAssertEqualObjects([NSSet setWithArray:[descriptor.parameters valueForKey:@"key"]],
							  [self declaredKeysOf:backend selector:@selector(supportedParameterKeys)], @"%@", descriptor.identifier);
		XCTAssertGreaterThan(descriptor.outputs.count, (NSUInteger)0, @"%@ declares what it writes", descriptor.identifier);
	}
}

- (void)testEveryPortIsTypedFromTheCoreTable
{
	for (NFKModelDescriptor *descriptor in [NFKAppleModelDescriptors modelDescriptors]) {
		for (NFKModelPort *port in descriptor.inputs) {
			XCTAssertEqualObjects(port.valueType, NFKModelValueTypeForKey(port.key), @"%@ %@", descriptor.identifier, port.key);
		}
		for (NFKModelPort *port in descriptor.outputs) {
			XCTAssertNotNil(port.valueType, @"%@ %@", descriptor.identifier, port.key);
			XCTAssertEqualObjects(port.valueType, NFKModelValueTypeForOutputKey(port.key), @"%@ %@", descriptor.identifier, port.key);
		}
	}
}

- (void)testIdentifiersAreUniqueAndNameTheirBackends
{
	NSArray<NFKModelDescriptor *> *descriptors = [NFKAppleModelDescriptors modelDescriptors];
	NSArray<NSString *> *identifiers = [descriptors valueForKey:@"identifier"];
	XCTAssertEqual([NSSet setWithArray:identifiers].count, identifiers.count);
	XCTAssertEqualObjects(identifiers, NFKAppleModelDescriptors.registryNames);
	for (NFKModelDescriptor *descriptor in descriptors) {
		XCTAssertEqualObjects(descriptor.registryName, descriptor.identifier);
		XCTAssertEqualObjects(descriptor.defaultVariant.registryName, descriptor.identifier);
		id<NFKInferenceBackend> backend = [NFKAppleModelDescriptors backendNamed:descriptor.registryName];
		XCTAssertTrue([descriptor.identifier hasPrefix:backend.backendIdentifier], @"%@ builds %@",
					  descriptor.identifier, backend.backendIdentifier);
	}
	XCTAssertNil([NFKAppleModelDescriptors backendNamed:@"vision-coreml"], @"an app supplies that model");
	XCTAssertNil([NFKAppleModelDescriptors backendNamed:@"no-such-engine"]);
}

- (void)testEveryDescriptorIsAnAppleSystemModel
{
	for (NFKModelDescriptor *descriptor in [NFKAppleModelDescriptors modelDescriptors]) {
		XCTAssertEqualObjects(descriptor.engine, NFKModelEngineApple);
		XCTAssertGreaterThan(descriptor.tasks.count, (NSUInteger)0, @"%@", descriptor.identifier);
		XCTAssertEqual(descriptor.variants.count, (NSUInteger)1, @"%@", descriptor.identifier);
		XCTAssertTrue(descriptor.defaultVariant.isDefault);
		XCTAssertEqualObjects(descriptor.defaultVariant.source.kind, NFKModelSourceKindSystem);
		XCTAssertEqualObjects(descriptor.defaultVariant.source.layout, NFKModelSourceLayoutNone);
		XCTAssertNil(descriptor.defaultVariant.downloadByteCount);
		XCTAssertEqualObjects(descriptor.license.name, @"Apple system framework");
		XCTAssertNotNil(descriptor.requirements.minimumOSVersions[@"macOS"], @"%@", descriptor.identifier);
		XCTAssertFalse(descriptor.isStandIn);
		XCTAssertEqualObjects(descriptor.introducedVersion, @"0.4.0");
	}
}

- (void)testAProviderNamedOnADescriptorIsLinked
{
	for (NFKModelDescriptor *descriptor in [NFKAppleModelDescriptors modelDescriptors]) {
		if (descriptor.providerClassName == nil) {
			continue;
		}
		XCTAssertTrue([NFKDynamicBackend isProviderAvailable:descriptor.providerClassName], @"%@", descriptor.identifier);
	}
	NFKModelDescriptor *text = [self descriptorNamed:@"vision-text"];
	XCTAssertEqualObjects(text.providerClassName, @"NFKVisionTextProvider");
	XCTAssertEqualObjects(text.tasks.firstObject, NFKCapabilityTextRecognition);
}

- (void)testTheFloorsFollowEachFramework
{
	NSDictionary *coreFloor = @{ @"macOS": @"11.0", @"iOS": @"14.0", @"tvOS": @"14.0" };
	XCTAssertEqualObjects([self descriptorNamed:@"vision-text"].requirements.minimumOSVersions, coreFloor);
	NSDictionary *soundFloor = @{ @"macOS": @"12.0", @"iOS": @"15.0", @"tvOS": @"15.0" };
	XCTAssertEqualObjects([self descriptorNamed:@"sound-analysis"].requirements.minimumOSVersions, soundFloor);
	XCTAssertNil([self descriptorNamed:@"apple-speech"].requirements.minimumOSVersions[@"tvOS"],
				 @"the recognizer does not run on tvOS");
	NFKModelDescriptor *flow = [self descriptorNamed:@"videotoolbox-optical-flow"];
	XCTAssertEqualObjects(flow.requirements.minimumOSVersions[@"macOS"], @"15.4");
	XCTAssertNil(flow.requirements.minimumOSVersions[@"tvOS"]);
	XCTAssertTrue(flow.requirements.requiresAppleSilicon);
	XCTAssertFalse([self descriptorNamed:@"vision-text"].requirements.requiresAppleSilicon);
}

- (void)testTheSharedCatalogHoldsTheAppleDescriptors
{
	NFKModelCatalog *catalog = NFKModelCatalog.sharedCatalog;
	for (NFKModelDescriptor *descriptor in [NFKAppleModelDescriptors modelDescriptors]) {
		XCTAssertEqualObjects([catalog descriptorWithIdentifier:descriptor.identifier].engine, NFKModelEngineApple,
							  @"%@", descriptor.identifier);
	}
	NSArray *recognizers = [[catalog descriptorsForTask:NFKModelTaskTextRecognition] valueForKey:@"identifier"];
	XCTAssertTrue([recognizers containsObject:@"vision-text"]);
	XCTAssertEqual([catalog descriptorsInFamily:@"apple-videotoolbox"].count, (NSUInteger)3);
}

- (void)testTheDescriptorsRoundTripThroughAManifest
{
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	[catalog addDescriptors:[NFKAppleModelDescriptors modelDescriptors]];
	NSError *error = nil;
	NSData *manifest = [catalog manifestDataWithError:&error];
	XCTAssertNotNil(manifest, @"%@", error);
	NFKModelCatalog *read = [[NFKModelCatalog alloc] init];
	XCTAssertTrue([read addDescriptorsFromManifestData:manifest error:&error], @"%@", error);
	XCTAssertEqualObjects([read manifestDataWithError:NULL], manifest);
}

#pragma mark Declared outputs

- (void)testABackendsOutputsFollowItsConfiguration
{
	NFKVisionFaceBackend *faces = [NFKVisionFaceBackend backend];
	NSSet *withLandmarks = [NSSet setWithObjects:NFKOutputDetections, NFKOutputPose, nil];
	XCTAssertEqualObjects(faces.supportedOutputKeys, withLandmarks);
	faces.detectsLandmarks = NO;
	XCTAssertEqualObjects(faces.supportedOutputKeys, [NSSet setWithObject:NFKOutputDetections]);

	NFKVisionTrackingBackend *tracker = [NFKVisionTrackingBackend backendWithKind:NFKVisionTrackingKindObject];
	XCTAssertEqualObjects(tracker.supportedOutputKeys, [NSSet setWithObject:NFKOutputDetections]);
	tracker.kind = NFKVisionTrackingKindTrajectory;
	XCTAssertEqualObjects(tracker.supportedOutputKeys, [NSSet setWithObject:NFKOutputStructured]);

	for (NSNumber *task in @[ @(NFKVideoToolboxTaskSuperResolution), @(NFKVideoToolboxTaskFrameInterpolation),
							  @(NFKVideoToolboxTaskOpticalFlow) ]) {
		NFKVideoToolboxBackend *processor = [NFKVideoToolboxBackend backendWithTask:task.integerValue];
		XCTAssertEqualObjects(processor.supportedOutputKeys, [NSSet setWithObject:NFKOutputImage], @"%@", task);
	}
}

- (nullable NFKModelDescriptor *)descriptorNamed:(NSString *)identifier
{
	for (NFKModelDescriptor *descriptor in [NFKAppleModelDescriptors modelDescriptors]) {
		if ([descriptor.identifier isEqualToString:identifier]) {
			return descriptor;
		}
	}
	XCTFail(@"no descriptor %@", identifier);
	return nil;
}

@end
