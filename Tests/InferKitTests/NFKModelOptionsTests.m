//
//  NFKModelOptionsTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

@interface NFKModelOptionsTests : XCTestCase
@end

@implementation NFKModelOptionsTests

- (NFKModelOption *)optionWithKey:(NSString *)key
							 kind:(NSString *)kind
							stage:(NSString *)stage
					 defaultValue:(nullable id)defaultValue
						  choices:(nullable NSArray<NFKModelOptionChoice *> *)choices
						  minimum:(nullable NSNumber *)minimum
						  maximum:(nullable NSNumber *)maximum
							 step:(nullable NSNumber *)step
{
	return [[NFKModelOption alloc] initWithKey:key kind:kind title:nil help:nil defaultValue:defaultValue minimum:minimum
									   maximum:maximum step:step choices:choices stage:stage group:nil extensions:nil];
}

// A chat model with load options of each kind and two request options.
- (NFKModelDescriptor *)chatModel
{
	NFKModelRequirements *unmeetable = [[NFKModelRequirements alloc] initWithMinimumOSVersions:nil requiresAppleSilicon:NO
																		minimumMemoryByteCount:nil
																			  requiredFeatures:@[ @"made-up-feature" ]
																				   unmetReason:nil extensions:nil];
	NFKModelOptionChoice *turbo = [[NFKModelOptionChoice alloc] initWithValue:@"turbo" title:@"Turbo" requirements:unmeetable
																	   memory:nil memoryByVariant:nil extensions:nil];
	NSArray<NFKModelOption *> *options = @[
		[self optionWithKey:NFKModelLoadOptionPrecision kind:NFKModelOptionKindChoice stage:NFKModelOptionStageLoad
			   defaultValue:NFKModelPrecisionCheckpoint
					choices:@[ [NFKModelOptionChoice choiceWithValue:NFKModelPrecisionCheckpoint title:@"Checkpoint"],
							   [NFKModelOptionChoice choiceWithValue:NFKModelPrecisionFloat32 title:@"Float32"], turbo ]
					minimum:nil maximum:nil step:nil],
		[self optionWithKey:NFKModelLoadOptionQuantizationBits kind:NFKModelOptionKindChoice stage:NFKModelOptionStageLoad
			   defaultValue:@0 choices:@[ [NFKModelOptionChoice choiceWithValue:@0 title:@"None"],
										  [NFKModelOptionChoice choiceWithValue:@8 title:@"8-bit"] ]
					minimum:nil maximum:nil step:nil],
		[self optionWithKey:NFKModelLoadOptionContextLength kind:NFKModelOptionKindInteger stage:NFKModelOptionStageLoad
			   defaultValue:@8192 choices:nil minimum:@512 maximum:@32768 step:@512],
		[self optionWithKey:@"width" kind:NFKModelOptionKindInteger stage:NFKModelOptionStageLoad defaultValue:@512
					choices:nil minimum:@256 maximum:@2048 step:@16],
		[self optionWithKey:@"adapter" kind:NFKModelOptionKindFileOrFolder stage:NFKModelOptionStageLoad defaultValue:nil
					choices:nil minimum:nil maximum:nil step:nil],
		[self optionWithKey:NFKModelLoadOptionDraftModel kind:NFKModelOptionKindModelReference stage:NFKModelOptionStageLoad
			   defaultValue:nil choices:nil minimum:nil maximum:nil step:nil],
		[self optionWithKey:@"steps" kind:NFKModelOptionKindInteger stage:NFKModelOptionStageRequest defaultValue:@20
					choices:nil minimum:@1 maximum:@50 step:nil],
		[self optionWithKey:NFKParameterTemperature kind:NFKModelOptionKindNumber stage:NFKModelOptionStageRequest
			   defaultValue:@0.7 choices:nil minimum:@0 maximum:@2 step:nil],
		[self optionWithKey:@"thinking" kind:NFKModelOptionKindBoolean stage:NFKModelOptionStageRequest defaultValue:nil
					choices:nil minimum:nil maximum:nil step:nil],
		[self optionWithKey:@"stops" kind:NFKModelOptionKindTextList stage:NFKModelOptionStageRequest defaultValue:nil
					choices:nil minimum:nil maximum:nil step:nil],
	];
	NFKModelMemoryModel *model = [[NFKModelMemoryModel alloc] initWithPrecisionOptionKey:NFKModelLoadOptionPrecision
															   quantizationBitsOptionKey:NFKModelLoadOptionQuantizationBits
														  quantizationGroupSizeOptionKey:nil
																  contextLengthOptionKey:NFKModelLoadOptionContextLength
																	  cacheBitsOptionKey:nil cacheGroupSizeOptionKey:nil
																		  draftOptionKey:NFKModelLoadOptionDraftModel
																	  draftBitsOptionKey:nil residencyOptionKey:nil
																			  cacheStage:nil draftStage:nil
																		 stagePrecisions:nil extensions:nil];
	NSArray *variants = @[ [NFKModelVariant variantWithIdentifier:@"small" displayName:@"Small"],
						   [NFKModelVariant variantWithIdentifier:@"large" displayName:@"Large"] ];
	return [[NFKModelDescriptor alloc] initWithIdentifier:@"example.chat" displayName:@"Example Chat" family:@"example"
												  summary:nil engine:NFKModelEnginePlugin registryName:nil
										providerClassName:nil tasks:nil inputModalities:nil outputModalities:nil inputs:nil
												  outputs:nil options:options variants:variants license:nil requirements:nil
											  residencies:nil memoryModel:model presets:nil standIn:NO
										introducedVersion:nil extensions:nil];
}

- (NSString *)messageFor:(NSDictionary *)options stage:(nullable NSString *)stage
{
	NSArray<NFKModelOptionProblem *> *problems = [[self chatModel] problemsWithOptions:options stage:stage];
	XCTAssertEqual(problems.count, 1u, @"%@", problems);
	return problems.firstObject.message;
}

- (NSString *)codeFor:(NSDictionary *)options stage:(nullable NSString *)stage
{
	return [[self chatModel] problemsWithOptions:options stage:stage].firstObject.code;
}

#pragma mark Resolving

- (void)testValuesConvertToTheirOptionsKinds
{
	NSError *error = nil;
	NSDictionary *resolved = [[self chatModel] resolvedOptions:@{ NFKModelLoadOptionContextLength: @"16384",
																  NFKModelLoadOptionQuantizationBits: @"8",
																  NFKParameterTemperature: @"1.5", @"thinking": @"true",
																  @"stops": @"END", @"steps": @30.0 }
														 stage:nil error:&error];
	XCTAssertNotNil(resolved, @"%@", error);
	XCTAssertEqualObjects(resolved[NFKModelLoadOptionContextLength], @16384);
	XCTAssertEqualObjects(resolved[NFKModelLoadOptionQuantizationBits], @8);
	XCTAssertEqualObjects(resolved[NFKParameterTemperature], @1.5);
	XCTAssertEqualObjects(resolved[@"thinking"], @YES);
	XCTAssertEqualObjects(resolved[@"stops"], (@[ @"END" ]));
	XCTAssertEqualObjects(resolved[@"steps"], @30);
}

- (void)testResolvingFillsEachDefaultOfTheStage
{
	NSDictionary *load = [[self chatModel] resolvedOptions:@{} stage:NFKModelOptionStageLoad error:NULL];
	XCTAssertEqualObjects(load, (@{ NFKModelLoadOptionVariant: @"small", NFKModelLoadOptionPrecision: NFKModelPrecisionCheckpoint,
									NFKModelLoadOptionQuantizationBits: @0, NFKModelLoadOptionContextLength: @8192,
									@"width": @512 }));
	NSDictionary *request = [[self chatModel] resolvedOptions:@{} stage:NFKModelOptionStageRequest error:NULL];
	XCTAssertEqualObjects(request, (@{ @"steps": @20, NFKParameterTemperature: @0.7 }));
}

- (void)testADescriptorWithSeveralVariantsDerivesAVariantOption
{
	NFKModelDescriptor *chat = [self chatModel];
	NFKModelOption *variant = [chat optionWithKey:NFKModelLoadOptionVariant];
	XCTAssertEqualObjects(variant.stage, NFKModelOptionStageLoad);
	XCTAssertEqualObjects([variant.choices valueForKey:@"value"], (@[ @"small", @"large" ]));
	XCTAssertEqualObjects(variant.defaultValue, @"small");
	XCTAssertEqualObjects(chat.loadOptions.firstObject.key, NFKModelLoadOptionVariant);
	XCTAssertEqual(chat.requestOptions.count, 4u);
	XCTAssertFalse([[chat.dictionaryRepresentation[@"options"] valueForKey:@"key"] containsObject:NFKModelLoadOptionVariant],
				   @"the derived option is not written");
	XCTAssertEqual([chat.dictionaryRepresentation[@"options"] count], chat.options.count);
}

- (void)testAnInvalidValueFailsWithItsProblems
{
	NSError *error = nil;
	NSDictionary *invalid = @{ @"steps": @0, @"nope": @1 };
	XCTAssertNil([[self chatModel] resolvedOptions:invalid stage:nil error:&error]);
	XCTAssertEqualObjects(error.domain, NFKModelCatalogErrorDomain);
	XCTAssertEqual(error.code, NFKModelCatalogErrorInvalidOptions);
	NSArray<NFKModelOptionProblem *> *problems = error.userInfo[NFKModelOptionProblemsKey];
	XCTAssertEqualObjects([problems valueForKey:@"code"], (@[ NFKModelOptionProblemUnknownKey, NFKModelOptionProblemBelowMinimum ]));
	XCTAssertEqualObjects(error.localizedDescription, @"example.chat has no option \"nope\".");
	XCTAssertEqualObjects(problems[1].dictionaryRepresentation,
						  (@{ @"key": @"steps", @"code": NFKModelOptionProblemBelowMinimum,
							  @"message": @"\"steps\" takes 1 or more, and 0 is less.", @"value": @0 }));
}

#pragma mark Problem messages

- (void)testEachProblemHasItsMessage
{
	XCTAssertEqualObjects([self messageFor:@{ @"steps": @"many" } stage:nil],
						  @"\"steps\" takes an integer, and \"many\" is not one.");
	XCTAssertEqualObjects([self messageFor:@{ @"steps": @80 } stage:nil], @"\"steps\" takes 50 or less, and 80 is more.");
	XCTAssertEqualObjects([self messageFor:@{ @"width": @260 } stage:nil],
						  @"\"width\" takes steps of 16 from 256, and 260 is between them.");
	XCTAssertEqualObjects([self messageFor:@{ NFKModelLoadOptionPrecision: @"int2" } stage:nil],
						  @"\"precision\" takes one of \"checkpoint\", \"float32\", \"turbo\", and \"int2\" is none of them.");
	XCTAssertEqualObjects([self messageFor:@{ NFKModelLoadOptionPrecision: @"turbo" } stage:nil],
						  @"\"precision\" cannot be \"turbo\" on this machine. This needs the made-up-feature feature.");
	XCTAssertEqualObjects([self messageFor:@{ NFKModelLoadOptionDraftModel: @"ghost" } stage:nil],
						  @"\"draftModel\" names \"ghost\", which is not in the catalog.");
	XCTAssertEqualObjects([self messageFor:@{ @"adapter": @"/nonexistent/adapter.safetensors" } stage:nil],
						  @"\"adapter\" names \"/nonexistent/adapter.safetensors\", which does not exist.");
	XCTAssertEqualObjects([self messageFor:@{ @"thinking": @"maybe" } stage:nil],
						  @"\"thinking\" takes a boolean, and \"maybe\" is not one.");
}

- (void)testEachProblemHasItsCode
{
	XCTAssertEqualObjects([self codeFor:@{ @"steps": @"many" } stage:nil], NFKModelOptionProblemWrongType);
	XCTAssertEqualObjects([self codeFor:@{ @"steps": @80 } stage:nil], NFKModelOptionProblemAboveMaximum);
	XCTAssertEqualObjects([self codeFor:@{ @"width": @260 } stage:nil], NFKModelOptionProblemOffStep);
	XCTAssertEqualObjects([self codeFor:@{ NFKModelLoadOptionPrecision: @"int2" } stage:nil], NFKModelOptionProblemNotAChoice);
	XCTAssertEqualObjects([self codeFor:@{ NFKModelLoadOptionPrecision: @"turbo" } stage:nil], NFKModelOptionProblemUnavailable);
	XCTAssertEqualObjects([self codeFor:@{ NFKModelLoadOptionDraftModel: @"ghost" } stage:nil], NFKModelOptionProblemUnknownModel);
	XCTAssertEqualObjects([self codeFor:@{ @"adapter": @"/nonexistent/a" } stage:nil], NFKModelOptionProblemMissingFile);
}

- (void)testAnOptionOfTheOtherStageIsAWrongStageProblem
{
	XCTAssertEqualObjects([self messageFor:@{ NFKModelLoadOptionPrecision: NFKModelPrecisionFloat32 }
									 stage:NFKModelOptionStageRequest],
						  @"\"precision\" is set when the model loads.");
	XCTAssertEqualObjects([self messageFor:@{ NFKParameterTemperature: @0.5 } stage:NFKModelOptionStageLoad],
						  @"\"temperature\" is set on each request.");
	XCTAssertEqualObjects([self codeFor:@{ NFKParameterTemperature: @0.5 } stage:NFKModelOptionStageLoad],
						  NFKModelOptionProblemWrongStage);
}

- (void)testARequestMayOverrideTheContextLength
{
	NSArray *problems = [[self chatModel] problemsWithOptions:@{ NFKModelLoadOptionContextLength: @4096 }
														stage:NFKModelOptionStageRequest];
	XCTAssertEqual(problems.count, 0u);
}

- (void)testTheMemoryBudgetOverrideIsALoadOptionLeftOutOfTheResult
{
	NSDictionary *resolved = [[self chatModel] resolvedOptions:@{ NFKModelLoadOptionIgnoreMemoryBudget: @YES }
														 stage:NFKModelOptionStageLoad error:NULL];
	XCTAssertNotNil(resolved);
	XCTAssertNil(resolved[NFKModelLoadOptionIgnoreMemoryBudget]);
	XCTAssertEqualObjects([self codeFor:@{ NFKModelLoadOptionIgnoreMemoryBudget: @YES } stage:NFKModelOptionStageRequest],
						  NFKModelOptionProblemWrongStage);
}

- (void)testTheCanonicalKeyIsTheSameForEquivalentOptions
{
	NFKModelDescriptor *chat = [self chatModel];
	NSString *spelled = [chat canonicalLoadOptionsKey:@{ NFKModelLoadOptionContextLength: @"8192",
														 NFKModelLoadOptionVariant: @"small" }];
	XCTAssertEqualObjects(spelled, [chat canonicalLoadOptionsKey:nil]);
	XCTAssertEqualObjects(spelled, @"{\"contextLength\":8192,\"precision\":\"checkpoint\",\"quantizationBits\":0,\"variant\":\"small\",\"width\":512}");
	XCTAssertNotEqualObjects(spelled, [chat canonicalLoadOptionsKey:@{ NFKModelLoadOptionVariant: @"large" }]);
	XCTAssertNil([chat canonicalLoadOptionsKey:@{ @"width": @3 }]);
}

#pragma mark Manifest fields

- (void)testTheManifestReadsOptionsOnlyFromTheOptionsField
{
	NSArray *seed = @[ @{ @"key": NFKParameterSeed, @"kind": NFKModelOptionKindInteger } ];
	NSDictionary *named = @{ @"id": @"m", @"family": @"f", @"engine": NFKModelEnginePlugin, @"parameters": seed };
	NFKModelDescriptor *descriptor = [NFKModelDescriptor descriptorWithDictionary:named error:NULL];
	XCTAssertEqual(descriptor.options.count, 0u);
	XCTAssertEqualObjects(descriptor.extensions[@"parameters"], seed);
	XCTAssertEqualObjects(descriptor.dictionaryRepresentation[@"parameters"], seed);

	NSDictionary *options = @{ @"id": @"m", @"family": @"f", @"engine": NFKModelEnginePlugin, @"options": seed };
	descriptor = [NFKModelDescriptor descriptorWithDictionary:options error:NULL];
	XCTAssertEqualObjects(descriptor.options.firstObject.key, NFKParameterSeed);
	XCTAssertNotNil(descriptor.dictionaryRepresentation[@"options"]);
}

- (void)testAnOptionCarriesItsRequirementsPropertyAndLinearMemory
{
	NSDictionary *dictionary = @{ @"key": NFKModelLoadOptionComputeUnits, @"kind": NFKModelOptionKindChoice,
								  @"stage": NFKModelOptionStageLoad, @"property": @"computeUnits",
								  @"memoryBytesPerUnit": @4096,
								  @"requires": @{ @"minimumOS": @{ @"macOS": @"14.0" }, @"requiresAppleSilicon": @NO,
												  @"requiredFeatures": @[ NFKModelFeatureNeuralAccelerators ],
												  @"unmetReason": @"Needs an M5." },
								  @"choices": @[ @{ @"value": @0, @"title": @"CPU",
													@"requires": @{ @"minimumOS": @{}, @"requiresAppleSilicon": @YES },
													@"memory": @{ @"residentBytes": @10, @"peakBytes": @12 },
													@"memoryByVariant": @{ @"large": @{ @"residentBytes": @20, @"peakBytes": @24 } } } ] };
	NSError *error = nil;
	NFKModelOption *option = [NFKModelOption optionWithDictionary:dictionary error:&error];
	XCTAssertNotNil(option, @"%@", error);
	XCTAssertEqualObjects(option.dictionaryRepresentation, dictionary);
	XCTAssertEqualObjects(option.propertyKeyPath, @"computeUnits");
	XCTAssertEqualObjects(option.memoryBytesPerUnit, @4096);
	XCTAssertEqualObjects(option.requirements.requiredFeatures, (@[ NFKModelFeatureNeuralAccelerators ]));
	NFKModelOptionChoice *cpu = option.choices.firstObject;
	XCTAssertEqual([cpu memoryForVariant:@"large"].peakByteCount, 24);
	XCTAssertEqual([cpu memoryForVariant:@"small"].peakByteCount, 12);
	XCTAssertTrue(cpu.requirements.requiresAppleSilicon);
}

#pragma mark Requirements and the hardware profile

- (NFKHardwareProfile *)intelMacOnMacOS14
{
	return [NFKHardwareProfile profileWithDictionary:@{ @"operatingSystem": @"macOS", @"operatingSystemVersion": @"14.6",
														 @"isAppleSilicon": @NO, @"hasNeuralAccelerators": @NO,
														 @"recommendedWorkingSetSize": @(16LL << 30) }];
}

- (void)testRequirementsNameTheFirstUnmetRequirement
{
	NFKHardwareProfile *profile = [self intelMacOnMacOS14];
	NFKModelRequirements *(^requirements)(NSDictionary *) = ^NFKModelRequirements *(NSDictionary *dictionary) {
		return [NFKModelRequirements requirementsWithDictionary:dictionary error:NULL];
	};
	XCTAssertNil([requirements(@{}) problemForHardwareProfile:profile]);
	XCTAssertNil([requirements(@{ @"minimumOS": @{ @"macOS": @"14.0" } }) problemForHardwareProfile:profile]);
	XCTAssertEqualObjects([requirements(@{ @"minimumOS": @{ @"macOS": @"14.10" } }) problemForHardwareProfile:profile],
						  @"This needs macOS 14.10 or later.");
	XCTAssertEqualObjects([requirements(@{ @"minimumOS": @{ @"iOS": @"17.0" } }) problemForHardwareProfile:profile],
						  @"This does not run on macOS.");
	XCTAssertEqualObjects([requirements(@{ @"requiresAppleSilicon": @YES }) problemForHardwareProfile:profile],
						  @"This needs Apple silicon.");
	XCTAssertEqualObjects([requirements(@{ @"requiredFeatures": @[ NFKModelFeatureNeuralAccelerators ] })
						   problemForHardwareProfile:profile], @"This needs a GPU with neural accelerators.");
	XCTAssertEqualObjects([requirements(@{ @"requiredFeatures": @[ NFKModelFeatureNeuralAccelerators ],
										   @"unmetReason": @"Needs an M5 or later." }) problemForHardwareProfile:profile],
						  @"Needs an M5 or later.");
}

- (void)testAHardwareProfileRoundTripsThroughItsSnapshot
{
	NFKHardwareProfile *current = NFKHardwareProfile.currentProfile;
	NSDictionary *snapshot = current.dictionaryRepresentation;
	XCTAssertTrue([NSJSONSerialization isValidJSONObject:snapshot]);
	XCTAssertEqualObjects(snapshot[@"operatingSystem"], @"macOS");
	NFKHardwareProfile *read = [NFKHardwareProfile profileWithDictionary:snapshot];
	XCTAssertEqualObjects(read.dictionaryRepresentation, snapshot);
	XCTAssertEqual(read.recommendedWorkingSetSize, current.recommendedWorkingSetSize);
	XCTAssertEqual(read.hasNeuralAccelerators, current.hasNeuralAccelerators);
}

- (void)testAProfileSnapshotSetsTheBudgetAndRulesOutUnmetChoices
{
	NFKHardwareProfile *profile = [self intelMacOnMacOS14];
	XCTAssertEqual([NFKModelMemoryBudget budgetForHardwareProfile:profile].byteCount,
				   (NSInteger)((double)(16LL << 30) * 0.85));
	NFKModelMemoryConstraints *constraints = [[self chatModel] constraintsForHardwareProfile:profile];
	NFKModelChoiceConstraint *turbo = [[constraints constraintForOptionKey:NFKModelLoadOptionPrecision]
									   constraintForChoiceValue:@"turbo"];
	XCTAssertEqual(turbo.status, NFKModelChoiceStatusRuledOut);
	XCTAssertEqualObjects(turbo.reason, @"Turbo is unavailable here. This needs the made-up-feature feature.");
}

@end
