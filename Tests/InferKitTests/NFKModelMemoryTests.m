//
//  NFKModelMemoryTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

static const NSInteger NFKTestGiB = (NSInteger)1 << 30;

@interface NFKModelMemoryTests : XCTestCase
@end

@implementation NFKModelMemoryTests

#pragma mark Fixtures

- (NFKModelOptionChoice *)choice:(id)value title:(NSString *)title
{
	return [NFKModelOptionChoice choiceWithValue:value title:title];
}

- (NFKModelOption *)loadOptionWithKey:(NSString *)key
								 kind:(NSString *)kind
								title:(NSString *)title
						 defaultValue:(id)defaultValue
							  choices:(nullable NSArray<NFKModelOptionChoice *> *)choices
							  minimum:(nullable NSNumber *)minimum
							  maximum:(nullable NSNumber *)maximum
								 step:(nullable NSNumber *)step
{
	return [[NFKModelOption alloc] initWithKey:key kind:kind title:title help:nil defaultValue:defaultValue minimum:minimum
									   maximum:maximum step:step choices:choices stage:NFKModelOptionStageLoad group:nil
									extensions:nil];
}

// A one-stage decoder: 1e9 parameters stored at bfloat16, 9e8 of them quantizable, a growing cache of
// 100,000 bytes per position at 16 bits, and 1,000,000 bytes of bounded cache.
- (NFKModelDescriptor *)decoderWithOverhead:(NSInteger)overhead
{
	NFKModelStageFacts *stage = [[NFKModelStageFacts alloc] initWithIdentifier:@"decoder" title:@"Decoder"
														  byteCountByPrecision:@{ NFKModelPrecisionCheckpoint: @2000000000,
																				  NFKModelPrecisionFloat32: @4000000000 }
													   quantizableElementCount:900000000 pageableByteCountByPrecision:nil
																	layerCount:0 largestLayerByteCountByPrecision:nil
															  widenedPrecision:nil extensions:nil];
	NFKModelMemoryFacts *facts = [[NFKModelMemoryFacts alloc] initWithStages:@[ stage ] keyValueByteCountPerToken:@100000
													boundedKeyValueByteCount:@1000000 maximumContextLength:@32768
															  vocabularySize:@151936 loadOverheadByteCount:overhead
																	   basis:nil extensions:nil];
	NFKModelVariant *variant = [[NFKModelVariant alloc] initWithIdentifier:@"1b" displayName:@"1B" isDefault:YES
															  registryName:nil source:nil downloadByteCount:nil
														 residentByteCount:nil peakByteCount:nil
															parameterCount:@1000000000 precision:NFKModelPrecisionBFloat16
													minimumMemoryByteCount:nil license:nil memoryFacts:facts
																extensions:nil];
	NSArray<NFKModelOptionChoice *> *bits = @[ [self choice:@0 title:@"None"], [self choice:@8 title:@"8-bit"],
											   [self choice:@4 title:@"4-bit"] ];
	NSArray<NFKModelOption *> *options = @[
		[self loadOptionWithKey:NFKModelLoadOptionPrecision kind:NFKModelOptionKindChoice title:@"Precision"
				   defaultValue:NFKModelPrecisionCheckpoint
						choices:@[ [self choice:NFKModelPrecisionCheckpoint title:@"Checkpoint"],
								   [self choice:NFKModelPrecisionFloat32 title:@"Float32"] ]
						minimum:nil maximum:nil step:nil],
		[self loadOptionWithKey:NFKModelLoadOptionQuantizationBits kind:NFKModelOptionKindChoice title:@"Quantization"
				   defaultValue:@0 choices:bits minimum:nil maximum:nil step:nil],
		[self loadOptionWithKey:NFKModelLoadOptionContextLength kind:NFKModelOptionKindInteger title:@"Context length"
				   defaultValue:@8192 choices:nil minimum:@512 maximum:@32768 step:@512],
		[self loadOptionWithKey:NFKModelLoadOptionCacheBits kind:NFKModelOptionKindChoice title:@"Cache quantization"
				   defaultValue:@0 choices:bits minimum:nil maximum:nil step:nil],
	];
	NFKModelMemoryModel *model = [[NFKModelMemoryModel alloc] initWithPrecisionOptionKey:NFKModelLoadOptionPrecision
															   quantizationBitsOptionKey:NFKModelLoadOptionQuantizationBits
														  quantizationGroupSizeOptionKey:nil
																  contextLengthOptionKey:NFKModelLoadOptionContextLength
																	  cacheBitsOptionKey:NFKModelLoadOptionCacheBits
																 cacheGroupSizeOptionKey:nil draftOptionKey:nil
																	  draftBitsOptionKey:nil residencyOptionKey:nil
																			  cacheStage:nil draftStage:nil
																		 stagePrecisions:nil extensions:nil];
	return [[NFKModelDescriptor alloc] initWithIdentifier:@"test.decoder" displayName:@"Test Decoder" family:@"test"
												  summary:nil engine:NFKModelEnginePlugin registryName:nil
										providerClassName:nil tasks:@[ NFKModelTaskTextGeneration ] inputModalities:nil
										 outputModalities:nil inputs:nil outputs:nil options:options variants:@[ variant ]
												  license:nil requirements:nil residencies:nil memoryModel:model presets:nil
												  standIn:NO introducedVersion:nil extensions:nil];
}

// One stage of 41 GiB whose routed experts are 32 GiB, with a residency option.
- (NFKModelDescriptor *)mixtureOfExperts
{
	NSInteger stored = 41 * NFKTestGiB;
	NFKModelStageFacts *stage = [[NFKModelStageFacts alloc] initWithIdentifier:@"decoder" title:nil
														  byteCountByPrecision:@{ NFKModelPrecisionCheckpoint: @(stored) }
													   quantizableElementCount:0
												  pageableByteCountByPrecision:@{ NFKModelPrecisionCheckpoint: @(32 * NFKTestGiB) }
																	layerCount:0 largestLayerByteCountByPrecision:nil
															  widenedPrecision:nil extensions:nil];
	NFKModelMemoryFacts *facts = [[NFKModelMemoryFacts alloc] initWithStages:@[ stage ] keyValueByteCountPerToken:nil
													boundedKeyValueByteCount:nil maximumContextLength:nil vocabularySize:nil
													   loadOverheadByteCount:0 basis:nil extensions:nil];
	NFKModelVariant *variant = [[NFKModelVariant alloc] initWithIdentifier:@"base" displayName:nil isDefault:YES
															  registryName:nil source:nil downloadByteCount:nil
														 residentByteCount:nil peakByteCount:nil parameterCount:nil
																 precision:NFKModelPrecisionBFloat16 minimumMemoryByteCount:nil
																   license:nil memoryFacts:facts extensions:nil];
	NFKModelOption *residency = [self loadOptionWithKey:NFKModelLoadOptionResidency kind:NFKModelOptionKindChoice
												  title:@"Residency" defaultValue:NFKModelResidencyAutomatic
												choices:@[ [self choice:NFKModelResidencyAutomatic title:@"Automatic"],
														   [self choice:NFKModelResidencyResident title:@"Resident"],
														   [self choice:NFKModelResidencyStaged title:@"Staged"],
														   [self choice:NFKModelResidencyPaged title:@"Paged"] ]
												minimum:nil maximum:nil step:nil];
	NFKModelMemoryModel *model = [[NFKModelMemoryModel alloc] initWithPrecisionOptionKey:nil quantizationBitsOptionKey:nil
														  quantizationGroupSizeOptionKey:nil contextLengthOptionKey:nil
																	  cacheBitsOptionKey:nil cacheGroupSizeOptionKey:nil
																		  draftOptionKey:nil draftBitsOptionKey:nil
																	  residencyOptionKey:NFKModelLoadOptionResidency
																			  cacheStage:nil draftStage:nil
																		 stagePrecisions:nil extensions:nil];
	return [[NFKModelDescriptor alloc] initWithIdentifier:@"test.moe" displayName:nil family:@"test" summary:nil
												   engine:NFKModelEnginePlugin registryName:nil providerClassName:nil
													tasks:nil inputModalities:nil outputModalities:nil inputs:nil outputs:nil
												  options:@[ residency ] variants:@[ variant ] license:nil requirements:nil
											  residencies:nil memoryModel:model presets:nil standIn:NO
										introducedVersion:nil extensions:nil];
}

- (NFKModelMemoryEstimate *)estimateOf:(NFKModelDescriptor *)descriptor options:(NSDictionary *)options
{
	NSError *error = nil;
	NFKModelMemoryEstimate *estimate = [descriptor estimatedMemoryForOptions:options error:&error];
	XCTAssertNotNil(estimate, @"%@", error);
	return estimate;
}

#pragma mark Budget

- (void)testTheBudgetIsEightyFivePercentOfTheWorkingSetBesideAFourGiBReserve
{
	XCTAssertEqual(NFKModelMemoryBudget.workingSetFraction, 0.85);
	XCTAssertEqual(NFKModelMemoryBudget.reserveByteCount, 4 * NFKTestGiB);
	NFKModelMemoryBudget *budget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:32 * NFKTestGiB];
	XCTAssertEqual(budget.byteCount, 29205777612);
	XCTAssertTrue(budget.isKnown);
	XCTAssertTrue([budget holdsByteCount:29205777612 - 4 * NFKTestGiB]);
	XCTAssertFalse([budget holdsByteCount:29205777612 - 4 * NFKTestGiB + 1]);
	XCTAssertEqualObjects([NFKModelMemoryBudget formattedByteCount:budget.byteCount], @"27.2 GiB");
}

- (void)testAnUnknownBudgetAdmitsEveryStageAndHoldsNone
{
	NFKModelMemoryBudget *budget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:0];
	XCTAssertFalse(budget.isKnown);
	XCTAssertEqual(budget.byteCount, 0);
	XCTAssertTrue([budget admitsByteCount:1000 * NFKTestGiB]);
	XCTAssertFalse([budget holdsByteCount:1]);
}

- (void)testTheLeastWorkingSetAddsTheReserveAndDividesByTheFraction
{
	XCTAssertEqual([NFKModelMemoryBudget leastWorkingSetHoldingByteCount:0], 5052902702);
	XCTAssertEqual([NFKModelMemoryBudget leastWorkingSetHoldingByteCount:4 * NFKTestGiB], 10105805403);
}

- (void)testABudgetRoundTripsThroughItsDictionary
{
	NFKModelMemoryBudget *budget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:32 * NFKTestGiB];
	NSDictionary *dictionary = budget.dictionaryRepresentation;
	XCTAssertEqualObjects(dictionary[@"budgetBytes"], @29205777612);
	XCTAssertEqualObjects(dictionary[@"reserveBytes"], @(4 * NFKTestGiB));
	XCTAssertEqualObjects([NFKModelMemoryBudget budgetWithDictionary:dictionary error:NULL], budget);
	NSError *error = nil;
	XCTAssertNil([NFKModelMemoryBudget budgetWithDictionary:@{} error:&error]);
	XCTAssertEqualObjects(error.localizedDescription, @"budget.workingSetBytes is missing");
}

#pragma mark Estimate arithmetic

- (void)testTheEstimateFollowsTheArithmeticOfEachOption
{
	NFKModelDescriptor *decoder = [self decoderWithOverhead:0];
	// Each row: the options, the weights, and the key-value cache.
	NSArray<NSArray *> *rows = @[
		// 2e9 weights; (100,000 × 8192 + 1,000,000) bytes of cache at 16 bits.
		@[ @{}, @2000000000, @820200000 ],
		// float32 doubles the weights and the cache's elements.
		@[ @{ NFKModelLoadOptionPrecision: NFKModelPrecisionFloat32 }, @4000000000, @1640400000 ],
		// 4-bit: 2e9 − 9e8 × 2 + 9e8 × (4/8 + 4/64).
		@[ @{ NFKModelLoadOptionQuantizationBits: @4 }, @706250000, @820200000 ],
		// 8-bit: 2e9 − 9e8 × 2 + 9e8 × (8/8 + 4/64).
		@[ @{ NFKModelLoadOptionQuantizationBits: @8 }, @1156250000, @820200000 ],
		// An 8-bit cache: 820,200,000 / 2 × (8/8 + 4/64).
		@[ @{ NFKModelLoadOptionCacheBits: @8 }, @2000000000, @435731250 ],
		// A 4-bit cache: 820,200,000 / 2 × (4/8 + 4/64).
		@[ @{ NFKModelLoadOptionCacheBits: @4 }, @2000000000, @230681250 ],
		// 16384 positions.
		@[ @{ NFKModelLoadOptionContextLength: @16384 }, @2000000000, @1639400000 ],
		// A string converts the way a request body carries it.
		@[ @{ NFKModelLoadOptionContextLength: @"16384" }, @2000000000, @1639400000 ],
	];
	for (NSArray *row in rows) {
		NFKModelMemoryEstimate *estimate = [self estimateOf:decoder options:row[0]];
		NSInteger weights = [row[1] integerValue], cache = [row[2] integerValue];
		XCTAssertEqualObjects(estimate.components[NFKModelMemoryComponentWeights], row[1], @"%@", row[0]);
		XCTAssertEqualObjects(estimate.components[NFKModelMemoryComponentKeyValueCache], row[2], @"%@", row[0]);
		XCTAssertEqual(estimate.residentByteCount, weights + cache, @"%@", row[0]);
		XCTAssertEqual(estimate.peakByteCount, weights + cache, @"%@", row[0]);
		XCTAssertTrue(estimate.fits, @"an estimate without a budget fits");
		XCTAssertEqualObjects(estimate.basis, NFKModelMemoryBasisMeasured);
	}
}

- (void)testThePeakAddsTheLoadOverhead
{
	NFKModelMemoryEstimate *estimate = [self estimateOf:[self decoderWithOverhead:100000000] options:@{}];
	XCTAssertEqual(estimate.residentByteCount, 2820200000);
	XCTAssertEqual(estimate.peakByteCount, 2920200000);
	XCTAssertEqualObjects(estimate.components[NFKModelMemoryComponentLoadOverhead], @100000000);
	XCTAssertEqual(estimate.minimumWorkingSetByteCount, [NFKModelMemoryBudget leastWorkingSetHoldingByteCount:2920200000]);
}

- (void)testAnEstimateWithoutABudgetIsStagedAndOneWithABudgetIsResident
{
	NFKModelDescriptor *decoder = [self decoderWithOverhead:0];
	XCTAssertEqualObjects([self estimateOf:decoder options:@{}].residency, NFKModelResidencyStaged);
	NFKModelMemoryBudget *budget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:64 * NFKTestGiB];
	NFKModelMemoryEstimate *estimate = [decoder estimatedMemoryForOptions:@{} memoryBudget:budget error:NULL];
	XCTAssertEqualObjects(estimate.requestedResidency, NFKModelResidencyAutomatic);
	XCTAssertEqualObjects(estimate.residency, NFKModelResidencyResident);
	XCTAssertEqualObjects(estimate.stageByteCounts, (@{ @"decoder": @2820200000 }));
}

- (void)testAnEstimateThatDoesNotFitSaysWhy
{
	NFKModelMemoryBudget *budget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:4 * NFKTestGiB];
	NFKModelMemoryEstimate *estimate = [[self decoderWithOverhead:0] estimatedMemoryForOptions:@{} memoryBudget:budget
																						 error:NULL];
	XCTAssertFalse(estimate.fits);
	XCTAssertEqualObjects(estimate.reason, @"The model needs 2.6 GiB plus a 4.0 GiB reserve, against a 3.4 GiB budget.");
}

- (void)testAnEstimateRefusesOptionsThatDoNotResolve
{
	NSError *error = nil;
	XCTAssertNil([[self decoderWithOverhead:0] estimatedMemoryForOptions:@{ NFKModelLoadOptionContextLength: @100 }
																   error:&error]);
	XCTAssertEqual(error.code, NFKModelCatalogErrorInvalidOptions);
}

- (void)testAVariantWithoutFactsEstimatesItsDeclaredBytes
{
	NFKModelVariant *variant = [[NFKModelVariant alloc] initWithIdentifier:@"v" displayName:nil isDefault:YES
															  registryName:nil source:nil downloadByteCount:nil
														 residentByteCount:@1000 peakByteCount:@1500 parameterCount:nil
																 precision:nil minimumMemoryByteCount:nil extensions:nil];
	NFKModelDescriptor *declared = [[NFKModelDescriptor alloc] initWithIdentifier:@"d" displayName:nil family:@"f"
																		  summary:nil engine:NFKModelEnginePlugin
																	 registryName:nil providerClassName:nil tasks:nil
																  inputModalities:nil outputModalities:nil inputs:nil
																		  outputs:nil options:nil variants:@[ variant ]
																		  license:nil requirements:nil residencies:nil
																		  standIn:NO introducedVersion:nil extensions:nil];
	NFKModelMemoryEstimate *estimate = [self estimateOf:declared options:@{}];
	XCTAssertEqual(estimate.residentByteCount, 1000);
	XCTAssertEqual(estimate.peakByteCount, 1500);
	XCTAssertEqualObjects(estimate.basis, NFKModelMemoryBasisDeclared);
}

- (void)testADraftModelAddsItsDraftStageAndItsOwnCacheToTheCacheStage
{
	NFKModelStageFacts *draftStage = [NFKModelStageFacts stageWithIdentifier:@"decoder"
														byteCountByPrecision:@{ NFKModelPrecisionCheckpoint: @200000000 }];
	NFKModelMemoryFacts *draftFacts = [[NFKModelMemoryFacts alloc] initWithStages:@[ draftStage ] keyValueByteCountPerToken:@10000
														 boundedKeyValueByteCount:nil maximumContextLength:nil
																   vocabularySize:@151936 loadOverheadByteCount:0 basis:nil
																	   extensions:nil];
	NFKModelVariant *draftVariant = [[NFKModelVariant alloc] initWithIdentifier:@"small" displayName:nil isDefault:YES
																   registryName:nil source:nil downloadByteCount:nil
															  residentByteCount:nil peakByteCount:nil
																 parameterCount:@100000000 precision:NFKModelPrecisionBFloat16
														 minimumMemoryByteCount:nil license:nil memoryFacts:draftFacts
																	 extensions:nil];
	NFKModelDescriptor *draft = [[NFKModelDescriptor alloc] initWithIdentifier:@"test.draft" displayName:nil family:@"test"
																	   summary:nil engine:NFKModelEnginePlugin registryName:nil
															 providerClassName:nil tasks:nil inputModalities:nil
															  outputModalities:nil inputs:nil outputs:nil options:nil
																	  variants:@[ draftVariant ] license:nil requirements:nil
																   residencies:nil standIn:NO introducedVersion:nil
																	extensions:nil];
	NFKModelDescriptor *decoder = [self decoderWithDraftOption];
	[NFKModelCatalog.sharedCatalog addDescriptor:draft];
	[self addTeardownBlock:^{
		[NFKModelCatalog.sharedCatalog removeDescriptorWithIdentifier:@"test.draft"];
	}];

	XCTAssertEqualObjects([[decoder draftDescriptorsForVariant:nil inCatalog:NFKModelCatalog.sharedCatalog]
						   valueForKey:@"identifier"], (@[ @"test.draft" ]));
	// The draft's weights, and its own cache at the run's 8192 positions: 10,000 × 8192 bytes at 16 bits.
	NFKModelMemoryEstimate *estimate = [self estimateOf:decoder options:@{ NFKModelLoadOptionDraftModel: @"test.draft" }];
	XCTAssertEqualObjects(estimate.components[NFKModelMemoryComponentDraft], @281920000);
	XCTAssertEqual(estimate.residentByteCount, (NSInteger)2000000000 + 820200000 + 281920000);

	// An 8-bit cache quantizes the draft's cache too: 81,920,000 / 2 × (8/8 + 4/64).
	NFKModelMemoryEstimate *quantized = [self estimateOf:decoder options:@{ NFKModelLoadOptionDraftModel: @"test.draft",
																			NFKModelLoadOptionCacheBits: @8 }];
	XCTAssertEqualObjects(quantized.components[NFKModelMemoryComponentDraft], @243520000);
	XCTAssertEqualObjects(quantized.components[NFKModelMemoryComponentKeyValueCache], @435731250);

	// A longer context grows both caches: the draft's at 16384 positions is 163,840,000 bytes.
	NFKModelMemoryEstimate *longer = [self estimateOf:decoder options:@{ NFKModelLoadOptionDraftModel: @"test.draft",
																		 NFKModelLoadOptionContextLength: @16384 }];
	XCTAssertEqualObjects(longer.components[NFKModelMemoryComponentDraft], @363840000);

	// The context length's bytes per position count the draft's cache beside the model's.
	NFKModelMemoryBudget *budget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:8 * NFKTestGiB];
	NSDictionary *fixed = @{ NFKModelLoadOptionDraftModel: @"test.draft", NFKModelLoadOptionPrecision: NFKModelPrecisionCheckpoint,
							 NFKModelLoadOptionQuantizationBits: @0, NFKModelLoadOptionCacheBits: @0 };
	NFKModelMemoryConstraints *constraints = [decoder constraintsForOptions:fixed memoryBudget:budget];
	// (7,301,444,403 − 4 GiB − 2,201,000,000) / 110,000 = 7,322, rounded down to a step of 512 from 512.
	XCTAssertEqualObjects([constraints constraintForOptionKey:NFKModelLoadOptionContextLength].maximumValue, @7168);
}

// The decoder with a draft-model option and no choices: its drafts are the compatible descriptors.
- (NFKModelDescriptor *)decoderWithDraftOption
{
	NFKModelDescriptor *decoder = [self decoderWithOverhead:0];
	NFKModelOption *draft = [[NFKModelOption alloc] initWithKey:NFKModelLoadOptionDraftModel
														   kind:NFKModelOptionKindModelReference title:@"Draft model"
														   help:nil defaultValue:nil minimum:nil maximum:nil step:nil
														choices:nil stage:NFKModelOptionStageLoad group:nil extensions:nil];
	NFKModelMemoryModel *model = decoder.memoryModel;
	NFKModelMemoryModel *withDraft = [[NFKModelMemoryModel alloc] initWithPrecisionOptionKey:model.precisionOptionKey
																   quantizationBitsOptionKey:model.quantizationBitsOptionKey
															  quantizationGroupSizeOptionKey:nil
																	  contextLengthOptionKey:model.contextLengthOptionKey
																		  cacheBitsOptionKey:model.cacheBitsOptionKey
																	 cacheGroupSizeOptionKey:nil
																			  draftOptionKey:NFKModelLoadOptionDraftModel
																		  draftBitsOptionKey:nil residencyOptionKey:nil
																				  cacheStage:nil draftStage:nil
																			 stagePrecisions:nil extensions:nil];
	return [[NFKModelDescriptor alloc] initWithIdentifier:decoder.identifier displayName:nil family:decoder.family
												  summary:nil engine:decoder.engine registryName:nil providerClassName:nil
													tasks:nil inputModalities:nil outputModalities:nil inputs:nil outputs:nil
												  options:[decoder.options arrayByAddingObject:draft] variants:decoder.variants
												  license:nil requirements:nil residencies:nil memoryModel:withDraft
												  presets:nil standIn:NO introducedVersion:nil extensions:nil];
}

#pragma mark Constraints

- (void)testAModelTooLargeToHoldWholeRequiresPaging
{
	NFKModelMemoryBudget *budget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:32 * NFKTestGiB];
	NFKModelMemoryConstraints *constraints = [[self mixtureOfExperts] constraintsForMemoryBudget:budget];
	XCTAssertTrue(constraints.anyCombinationFits);
	NFKModelOptionConstraint *residency = [constraints constraintForOptionKey:NFKModelLoadOptionResidency];
	XCTAssertEqualObjects(residency.requiredValue, NFKModelResidencyPaged);
	XCTAssertEqualObjects(residency.reason, @"The model needs 41.0 GiB fully resident plus a 4.0 GiB reserve, against a 27.2 GiB budget. Paged, it needs 16.1 GiB.");
	XCTAssertEqualObjects(residency.fullyResidentByteCount, @(41 * NFKTestGiB));
	XCTAssertEqualObjects(residency.budgetByteCount, @29205777612);
	XCTAssertEqualObjects(residency.techniqueByteCount, @17287243366);

	XCTAssertEqual([residency constraintForChoiceValue:NFKModelResidencyPaged].status, NFKModelChoiceStatusRequired);
	XCTAssertEqual([residency constraintForChoiceValue:NFKModelResidencyAutomatic].status, NFKModelChoiceStatusAllowed);
	NFKModelChoiceConstraint *resident = [residency constraintForChoiceValue:NFKModelResidencyResident];
	XCTAssertEqual(resident.status, NFKModelChoiceStatusRuledOut);
	XCTAssertEqualObjects(resident.reason, @"Resident needs at least 41.0 GiB plus a 4.0 GiB reserve, against a 27.2 GiB budget.");
	XCTAssertEqual([residency constraintForChoiceValue:NFKModelResidencyStaged].status, NFKModelChoiceStatusRuledOut);

	XCTAssertEqualObjects(constraints.fittingOptions[NFKModelLoadOptionResidency], NFKModelResidencyAutomatic);
	XCTAssertEqualObjects(constraints.fittingEstimate.residency, NFKModelResidencyPaged);
	XCTAssertEqual(constraints.fittingEstimate.residentByteCount, 17287243366);
	XCTAssertEqualObjects(constraints.fittingEstimate.components[NFKModelMemoryComponentExpertCache], @7623566950);
}

- (void)testAnIntegerOptionStatesTheLargestValueThatFits
{
	NFKModelMemoryBudget *budget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:8 * NFKTestGiB];
	NFKModelMemoryConstraints *constraints = [[self decoderWithOverhead:0]
											  constraintsForOptions:@{ NFKModelLoadOptionPrecision: NFKModelPrecisionCheckpoint,
																	   NFKModelLoadOptionQuantizationBits: @0,
																	   NFKModelLoadOptionCacheBits: @0 }
											  memoryBudget:budget];
	// (7,301,444,403 − 4 GiB − 2,001,000,000) / 100,000 = 10,054, rounded down to a step of 512 from 512.
	NFKModelOptionConstraint *context = [constraints constraintForOptionKey:NFKModelLoadOptionContextLength];
	XCTAssertEqualObjects(context.maximumValue, @9728);
	XCTAssertEqualObjects(context.reason, @"Context length fits up to 9728 against a 6.8 GiB budget.");
	XCTAssertEqualObjects(constraints.fittingOptions[NFKModelLoadOptionContextLength], @8192);
}

- (void)testAChoiceThatFitsNowhereIsRuledOut
{
	NFKModelMemoryBudget *budget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:8 * NFKTestGiB];
	// Unquantized, float32 holds 4e9 bytes of weights, which a 6.8 GiB budget does not hold beside the reserve.
	NFKModelMemoryConstraints *constraints = [[self decoderWithOverhead:0]
											  constraintsForOptions:@{ NFKModelLoadOptionQuantizationBits: @0,
																	   NFKModelLoadOptionCacheBits: @0 }
											  memoryBudget:budget];
	NFKModelOptionConstraint *precision = [constraints constraintForOptionKey:NFKModelLoadOptionPrecision];
	XCTAssertEqual([precision constraintForChoiceValue:NFKModelPrecisionCheckpoint].status, NFKModelChoiceStatusRequired);
	XCTAssertEqual([precision constraintForChoiceValue:NFKModelPrecisionFloat32].status, NFKModelChoiceStatusRuledOut);
	XCTAssertEqualObjects(precision.requiredValue, NFKModelPrecisionCheckpoint);
	XCTAssertEqualObjects(precision.reason, @"Only Checkpoint fits: it needs 1.9 GiB plus a 4.0 GiB reserve, against a 6.8 GiB budget.");

	// Quantized, float32 fits as well, so no precision is required.
	NFKModelMemoryConstraints *free = [[self decoderWithOverhead:0] constraintsForMemoryBudget:budget];
	NFKModelOptionConstraint *freePrecision = [free constraintForOptionKey:NFKModelLoadOptionPrecision];
	XCTAssertNil(freePrecision.requiredValue);
	XCTAssertEqual([freePrecision constraintForChoiceValue:NFKModelPrecisionFloat32].status, NFKModelChoiceStatusAllowed);
}

- (void)testNoCombinationFittingIsReported
{
	NFKModelMemoryBudget *budget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:4 * NFKTestGiB];
	NFKModelMemoryConstraints *constraints = [[self mixtureOfExperts] constraintsForMemoryBudget:budget];
	XCTAssertFalse(constraints.anyCombinationFits);
	XCTAssertNil(constraints.fittingOptions);
	XCTAssertTrue([constraints.reason hasPrefix:@"No combination of options fits: the smallest needs "]);
}

- (void)testConstraintsRoundTripThroughTheirDictionary
{
	NFKModelMemoryBudget *budget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:32 * NFKTestGiB];
	NFKModelMemoryConstraints *constraints = [[self mixtureOfExperts] constraintsForMemoryBudget:budget];
	NSDictionary *dictionary = constraints.dictionaryRepresentation;
	XCTAssertTrue([NSJSONSerialization isValidJSONObject:dictionary]);
	NSError *error = nil;
	NFKModelMemoryConstraints *read = [NFKModelMemoryConstraints constraintsWithDictionary:dictionary error:&error];
	XCTAssertNotNil(read, @"%@", error);
	XCTAssertEqualObjects(read.dictionaryRepresentation, dictionary);
	XCTAssertEqualObjects(dictionary[@"options"][NFKModelLoadOptionResidency][@"requiredValue"], NFKModelResidencyPaged);
}

- (void)testAnEstimateRoundTripsThroughItsDictionary
{
	NFKModelMemoryBudget *budget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:64 * NFKTestGiB];
	NFKModelMemoryEstimate *estimate = [[self decoderWithOverhead:5] estimatedMemoryForOptions:@{} memoryBudget:budget
																						 error:NULL];
	NSDictionary *dictionary = estimate.dictionaryRepresentation;
	XCTAssertTrue([NSJSONSerialization isValidJSONObject:dictionary]);
	XCTAssertEqualObjects([NFKModelMemoryEstimate estimateWithDictionary:dictionary error:NULL], estimate);
	XCTAssertEqualObjects(dictionary[@"options"][NFKModelLoadOptionContextLength], @8192);
}

#pragma mark The enumeration cap

// The decoder with seven memory-affecting choice options over five variants: 5 × 4 × 3 × 3 × 3 × 3 × 3
// = 4860 combinations, past the cap of 4096.
- (NFKModelDescriptor *)decoderWithManyCombinations
{
	NFKModelDescriptor *decoder = [self decoderWithOverhead:0];
	NFKModelVariant *base = decoder.defaultVariant;
	NSMutableArray<NFKModelVariant *> *variants = [NSMutableArray array];
	for (NSInteger index = 0; index < 5; index += 1) {
		[variants addObject:[[NFKModelVariant alloc] initWithIdentifier:[NSString stringWithFormat:@"v%ld", (long)index]
															displayName:nil isDefault:index == 0 registryName:nil source:nil
													  downloadByteCount:nil residentByteCount:nil peakByteCount:nil
														 parameterCount:base.parameterCount precision:base.precision
												 minimumMemoryByteCount:nil license:nil memoryFacts:base.memoryFacts
															 extensions:nil]];
	}
	NSArray<NFKModelOptionChoice *> *bits = @[ [self choice:@0 title:@"None"], [self choice:@8 title:@"8-bit"],
											   [self choice:@4 title:@"4-bit"] ];
	NSArray<NFKModelOptionChoice *> *groups = @[ [self choice:@32 title:@"32"], [self choice:@64 title:@"64"],
												 [self choice:@128 title:@"128"] ];
	NSArray<NFKModelOption *> *options = @[
		[self loadOptionWithKey:NFKModelLoadOptionPrecision kind:NFKModelOptionKindChoice title:@"Precision"
				   defaultValue:NFKModelPrecisionCheckpoint
						choices:@[ [self choice:NFKModelPrecisionCheckpoint title:@"Checkpoint"],
								   [self choice:NFKModelPrecisionFloat32 title:@"Float32"],
								   [self choice:NFKModelPrecisionFloat16 title:@"Float16"],
								   [self choice:NFKModelPrecisionBFloat16 title:@"BFloat16"] ]
						minimum:nil maximum:nil step:nil],
		[self loadOptionWithKey:NFKModelLoadOptionQuantizationBits kind:NFKModelOptionKindChoice title:@"Quantization"
				   defaultValue:@0 choices:bits minimum:nil maximum:nil step:nil],
		[self loadOptionWithKey:NFKModelLoadOptionQuantizationGroupSize kind:NFKModelOptionKindChoice title:@"Group"
				   defaultValue:@64 choices:groups minimum:nil maximum:nil step:nil],
		[self loadOptionWithKey:NFKModelLoadOptionContextLength kind:NFKModelOptionKindInteger title:@"Context length"
				   defaultValue:@8192 choices:nil minimum:@512 maximum:@32768 step:@512],
		[self loadOptionWithKey:NFKModelLoadOptionCacheBits kind:NFKModelOptionKindChoice title:@"Cache quantization"
				   defaultValue:@0 choices:bits minimum:nil maximum:nil step:nil],
		[self loadOptionWithKey:NFKModelLoadOptionCacheGroupSize kind:NFKModelOptionKindChoice title:@"Cache group"
				   defaultValue:@64 choices:groups minimum:nil maximum:nil step:nil],
		[self loadOptionWithKey:NFKModelLoadOptionResidency kind:NFKModelOptionKindChoice title:@"Residency"
				   defaultValue:NFKModelResidencyAutomatic
						choices:@[ [self choice:NFKModelResidencyAutomatic title:@"Automatic"],
								   [self choice:NFKModelResidencyResident title:@"Resident"],
								   [self choice:NFKModelResidencyStaged title:@"Staged"] ]
						minimum:nil maximum:nil step:nil],
	];
	NFKModelMemoryModel *model = [[NFKModelMemoryModel alloc] initWithPrecisionOptionKey:NFKModelLoadOptionPrecision
															   quantizationBitsOptionKey:NFKModelLoadOptionQuantizationBits
														  quantizationGroupSizeOptionKey:NFKModelLoadOptionQuantizationGroupSize
																  contextLengthOptionKey:NFKModelLoadOptionContextLength
																	  cacheBitsOptionKey:NFKModelLoadOptionCacheBits
																 cacheGroupSizeOptionKey:NFKModelLoadOptionCacheGroupSize
																		  draftOptionKey:nil draftBitsOptionKey:nil
																	  residencyOptionKey:NFKModelLoadOptionResidency
																			  cacheStage:nil draftStage:nil
																		 stagePrecisions:nil extensions:nil];
	return [[NFKModelDescriptor alloc] initWithIdentifier:@"test.many" displayName:nil family:@"test" summary:nil
												   engine:NFKModelEnginePlugin registryName:nil providerClassName:nil
													tasks:nil inputModalities:nil outputModalities:nil inputs:nil outputs:nil
												  options:options variants:variants license:nil requirements:nil
											  residencies:nil memoryModel:model presets:nil standIn:NO
										introducedVersion:nil extensions:nil];
}

- (void)testPastTheCapEachChoiceIsJudgedAloneAndTheResultIsNotExhaustive
{
	NFKModelMemoryBudget *budget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:8 * NFKTestGiB];
	NFKModelDescriptor *many = [self decoderWithManyCombinations];
	NFKModelMemoryConstraints *constraints = [many constraintsForMemoryBudget:budget];
	XCTAssertFalse(constraints.isExhaustive);
	NSDictionary *expectedCounts = @{ NFKModelLoadOptionVariant: @5, NFKModelLoadOptionPrecision: @4,
									  NFKModelLoadOptionQuantizationBits: @3, NFKModelLoadOptionQuantizationGroupSize: @3,
									  NFKModelLoadOptionCacheBits: @3, NFKModelLoadOptionCacheGroupSize: @3,
									  NFKModelLoadOptionResidency: @3 };
	for (NSString *key in expectedCounts) {
		XCTAssertEqual([constraints constraintForOptionKey:key].choices.count, [expectedCounts[key] unsignedIntegerValue],
					   @"every choice of %@ has a status", key);
	}
	// Alone, with every other option at its default, float32 holds 4e9 bytes of weights.
	NFKModelOptionConstraint *precision = [constraints constraintForOptionKey:NFKModelLoadOptionPrecision];
	XCTAssertEqual([precision constraintForChoiceValue:NFKModelPrecisionFloat32].status, NFKModelChoiceStatusRuledOut);
	XCTAssertEqual([precision constraintForChoiceValue:NFKModelPrecisionFloat16].status, NFKModelChoiceStatusAllowed);
	XCTAssertEqual([[constraints constraintForOptionKey:NFKModelLoadOptionVariant] constraintForChoiceValue:@"v4"].status,
				   NFKModelChoiceStatusAllowed);
	XCTAssertTrue(constraints.anyCombinationFits);

	NSDictionary *dictionary = constraints.dictionaryRepresentation;
	XCTAssertEqualObjects(dictionary[@"exhaustive"], @NO);
	XCTAssertFalse([NFKModelMemoryConstraints constraintsWithDictionary:dictionary error:NULL].isExhaustive);
}

- (void)testUnderTheCapTheEnumerationIsExhaustive
{
	NFKModelMemoryBudget *budget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:8 * NFKTestGiB];
	NFKModelMemoryConstraints *constraints = [[self decoderWithOverhead:0] constraintsForMemoryBudget:budget];
	XCTAssertTrue(constraints.isExhaustive);
	XCTAssertEqualObjects(constraints.dictionaryRepresentation[@"exhaustive"], @YES);
	NSMutableDictionary *withoutField = [constraints.dictionaryRepresentation mutableCopy];
	[withoutField removeObjectForKey:@"exhaustive"];
	XCTAssertTrue([NFKModelMemoryConstraints constraintsWithDictionary:withoutField error:NULL].isExhaustive);
}

#pragma mark Facts and figures

- (void)testFactsRoundTripThroughTheirDictionary
{
	NSDictionary *dictionary = @{
		@"stages": @[ @{ @"id": @"text_encoder", @"title": @"Text encoder",
						 @"bytesByPrecision": @{ @"checkpoint": @100, @"float32": @200 }, @"widenedPrecision": @"float32" },
					  @{ @"id": @"transformer", @"title": @"transformer", @"bytesByPrecision": @{ @"checkpoint": @1000 },
						 @"quantizableElements": @400, @"pageableBytesByPrecision": @{ @"checkpoint": @600 },
						 @"layerCount": @8, @"largestLayerBytesByPrecision": @{ @"checkpoint": @110 },
						 @"x-note": @"kept" } ],
		@"keyValueBytesPerToken": @4096, @"boundedKeyValueBytes": @0, @"maximumContextLength": @40960,
		@"vocabularySize": @151936, @"loadOverheadBytes": @0, @"basis": NFKModelMemoryBasisStoredDType,
	};
	NSError *error = nil;
	NFKModelMemoryFacts *facts = [NFKModelMemoryFacts factsWithDictionary:dictionary error:&error];
	XCTAssertNotNil(facts, @"%@", error);
	XCTAssertEqualObjects(facts.dictionaryRepresentation, dictionary);
	XCTAssertEqualObjects([facts stageWithIdentifier:@"transformer"].extensions, (@{ @"x-note": @"kept" }));
	XCTAssertEqual([facts stageWithIdentifier:@"transformer"].layerCount, 8);

	XCTAssertNil([NFKModelMemoryFacts factsWithDictionary:@{ @"stages": @[ @{ @"id": @"x" } ] } error:&error]);
	XCTAssertEqualObjects(error.localizedDescription, @"memoryFacts.stages[0].bytesByPrecision is missing");
}

- (void)testAMemoryModelRoundTripsThroughItsDictionary
{
	NSDictionary *dictionary = @{ @"precision": @"precision", @"quantizationBits": @"quantizationBits",
								  @"contextLength": @"contextLength", @"draft": @"draftModel", @"residency": @"residency",
								  @"cacheStage": @"decoder", @"draftStage": @"decoder",
								  @"stagePrecisions": @{ @"vision": @"float32" } };
	NFKModelMemoryModel *model = [NFKModelMemoryModel memoryModelWithDictionary:dictionary error:NULL];
	XCTAssertEqualObjects(model.dictionaryRepresentation, dictionary);
	XCTAssertEqualObjects(model.optionKeys, (@[ @"precision", @"quantizationBits", @"contextLength", @"draftModel",
											   @"residency" ]));
}

- (void)testAFigureRoundTripsAndAMissingPeakIsTheResidentBytes
{
	NFKModelMemoryFigure *figure = [NFKModelMemoryFigure figureWithDictionary:@{ @"residentBytes": @10 } error:NULL];
	XCTAssertEqual(figure.peakByteCount, 10);
	XCTAssertEqualObjects(figure.dictionaryRepresentation, (@{ @"residentBytes": @10, @"peakBytes": @10 }));
}

- (void)testAVariantCarriesItsFactsThroughTheManifest
{
	NFKModelDescriptor *decoder = [self decoderWithOverhead:0];
	NSDictionary *dictionary = decoder.dictionaryRepresentation;
	XCTAssertNotNil(dictionary[@"variants"][0][@"memoryFacts"]);
	XCTAssertNotNil(dictionary[@"memoryModel"]);
	NFKModelDescriptor *read = [NFKModelDescriptor descriptorWithDictionary:dictionary error:NULL];
	XCTAssertEqualObjects(read, decoder);
	XCTAssertEqualObjects(read.defaultVariant.memoryFacts, decoder.defaultVariant.memoryFacts);
}

- (void)testDerivingFiguresFillsEachMemoryChoice
{
	NFKModelDescriptor *derived = [[self decoderWithOverhead:0] descriptorByDerivingMemoryFigures];
	NFKModelOption *bits = [derived optionWithKey:NFKModelLoadOptionQuantizationBits];
	XCTAssertEqual([bits choiceWithValue:@4].memory.residentByteCount, 706250000 + 820200000);
	XCTAssertEqual([bits choiceWithValue:@0].memory.residentByteCount, 2820200000);
	XCTAssertEqualObjects([derived optionWithKey:NFKModelLoadOptionContextLength].memoryBytesPerUnit, @100000);
	XCTAssertEqualObjects(derived.defaultVariant.residentByteCount, @2820200000);
	XCTAssertEqualObjects(derived.defaultVariant.minimumMemoryByteCount,
						  @([NFKModelMemoryBudget leastWorkingSetHoldingByteCount:2820200000]));
}

- (void)testDerivingInACatalogFindsTheDraftThere
{
	NFKModelStageFacts *draftStage = [NFKModelStageFacts stageWithIdentifier:@"decoder"
														byteCountByPrecision:@{ NFKModelPrecisionCheckpoint: @200000000 }];
	NFKModelMemoryFacts *draftFacts = [[NFKModelMemoryFacts alloc] initWithStages:@[ draftStage ] keyValueByteCountPerToken:@10000
														 boundedKeyValueByteCount:nil maximumContextLength:nil
																   vocabularySize:@151936 loadOverheadByteCount:0 basis:nil
																	   extensions:nil];
	NFKModelVariant *draftVariant = [[NFKModelVariant alloc] initWithIdentifier:@"small" displayName:nil isDefault:YES
																   registryName:nil source:nil downloadByteCount:nil
															  residentByteCount:nil peakByteCount:nil
																 parameterCount:@100000000 precision:NFKModelPrecisionBFloat16
														 minimumMemoryByteCount:nil license:nil memoryFacts:draftFacts
																	 extensions:nil];
	NFKModelDescriptor *draft = [[NFKModelDescriptor alloc] initWithIdentifier:@"test.local-draft" displayName:nil
																		family:@"test" summary:nil engine:NFKModelEnginePlugin
																  registryName:nil providerClassName:nil tasks:nil
															   inputModalities:nil outputModalities:nil inputs:nil outputs:nil
																	   options:nil variants:@[ draftVariant ] license:nil
																  requirements:nil residencies:nil standIn:NO
															 introducedVersion:nil extensions:nil];
	NFKModelDescriptor *decoder = [self decoderWithDraftOption];
	NFKModelOption *listed = [[NFKModelOption alloc] initWithKey:NFKModelLoadOptionDraftModel
															kind:NFKModelOptionKindModelReference title:@"Draft model" help:nil
													defaultValue:nil minimum:nil maximum:nil step:nil
														 choices:@[ [self choice:@"test.local-draft" title:@"Local draft"] ]
														   stage:NFKModelOptionStageLoad group:nil extensions:nil];
	NSMutableArray<NFKModelOption *> *options = [decoder.options mutableCopy];
	[options replaceObjectAtIndex:options.count - 1 withObject:listed];
	NFKModelDescriptor *listing = [[NFKModelDescriptor alloc] initWithIdentifier:decoder.identifier displayName:nil
																		 family:decoder.family summary:nil
																		 engine:decoder.engine registryName:nil
															  providerClassName:nil tasks:nil inputModalities:nil
															   outputModalities:nil inputs:nil outputs:nil options:options
																	   variants:decoder.variants license:nil
																   requirements:nil residencies:nil
																	memoryModel:decoder.memoryModel presets:nil standIn:NO
															  introducedVersion:nil extensions:nil];
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	[catalog addDescriptors:@[ draft, listing ]];
	XCTAssertNil([NFKModelCatalog.sharedCatalog descriptorWithIdentifier:@"test.local-draft"]);

	NFKModelDescriptor *derived = [listing descriptorByDerivingMemoryFiguresInCatalog:catalog];
	NFKModelOptionChoice *choice = [[derived optionWithKey:NFKModelLoadOptionDraftModel] choiceWithValue:@"test.local-draft"];
	// The model at its defaults, and the draft's weights with its own cache at 8192 positions.
	XCTAssertEqual(choice.memory.residentByteCount, (NSInteger)2820200000 + 200000000 + 81920000);
}

- (void)testDerivingLeavesTheAutomaticResidencyWithoutAFigure
{
	NFKModelDescriptor *derived = [[self mixtureOfExperts] descriptorByDerivingMemoryFigures];
	NFKModelOption *residency = [derived optionWithKey:NFKModelLoadOptionResidency];
	XCTAssertNil([residency choiceWithValue:NFKModelResidencyAutomatic].memory);
	XCTAssertEqual([residency choiceWithValue:NFKModelResidencyResident].memory.residentByteCount, 41 * NFKTestGiB);
}

- (void)testAStatedFigureWinsOverTheDerivedOne
{
	NFKModelDescriptor *decoder = [self decoderWithOverhead:0];
	NFKModelMemoryFigure *stated = [[NFKModelMemoryFigure alloc] initWithResidentByteCount:7 peakByteCount:9];
	NFKModelOptionChoice *choice = [[NFKModelOptionChoice alloc] initWithValue:NFKModelPrecisionFloat32 title:@"Float32"
																  requirements:nil memory:stated memoryByVariant:nil
																	extensions:nil];
	NFKModelOption *precision = [[NFKModelOption alloc] initWithKey:NFKModelLoadOptionPrecision kind:NFKModelOptionKindChoice
															  title:nil help:nil defaultValue:NFKModelPrecisionCheckpoint
															minimum:nil maximum:nil step:nil
															choices:@[ [self choice:NFKModelPrecisionCheckpoint title:@"Checkpoint"], choice ]
															  stage:NFKModelOptionStageLoad group:nil extensions:nil];
	NSMutableArray *options = [decoder.options mutableCopy];
	options[0] = precision;
	NFKModelDescriptor *stating = [[NFKModelDescriptor alloc] initWithIdentifier:decoder.identifier displayName:nil
																		  family:decoder.family summary:nil
																		  engine:decoder.engine registryName:nil
															   providerClassName:nil tasks:nil inputModalities:nil
																outputModalities:nil inputs:nil outputs:nil options:options
																		variants:decoder.variants license:nil
																	requirements:nil residencies:nil
																	 memoryModel:decoder.memoryModel presets:nil standIn:NO
															   introducedVersion:nil extensions:nil];
	NFKModelDescriptor *derived = [stating descriptorByDerivingMemoryFigures];
	XCTAssertEqualObjects([[derived optionWithKey:NFKModelLoadOptionPrecision] choiceWithValue:NFKModelPrecisionFloat32].memory,
						  stated);
}

#pragma mark Residency planner

- (void)testThePlannerRefusesWithTheSwiftPlannersSentences
{
	NSInteger budget = 24 * NFKTestGiB;
	NFKModelStageFootprint *moe = [[NFKModelStageFootprint alloc] initWithByteCount:41 * NFKTestGiB
																  pageableByteCount:32 * NFKTestGiB
																   widenedByteCount:nil layerByteCounts:nil];
	NSError *error = nil;
	XCTAssertNil([NFKModelResidencyPlan planForStages:@[ moe ] residency:NFKModelResidencyResident budget:29205777612
												error:&error]);
	XCTAssertEqual(error.code, NFKModelCatalogErrorDoesNotFit);
	XCTAssertEqualObjects(error.localizedDescription,
						  @"the stages need 41.0 GiB together plus a 4.0 GiB reserve, against a 27.2 GiB working set; hold them staged or paged");

	NFKModelStageFootprint *heavyExperts = [[NFKModelStageFootprint alloc] initWithByteCount:40 * NFKTestGiB
																		   pageableByteCount:10 * NFKTestGiB
																			widenedByteCount:nil layerByteCounts:nil];
	XCTAssertNil([NFKModelResidencyPlan planForStages:@[ heavyExperts ] residency:NFKModelResidencyPaged budget:budget
												error:&error]);
	XCTAssertEqualObjects(error.localizedDescription,
						  @"the largest stage needs 30.0 GiB with its routed experts paged, plus a 4.0 GiB reserve, against a 24.0 GiB working set");

	NFKModelStageFootprint *dense = [[NFKModelStageFootprint alloc] initWithByteCount:30 * NFKTestGiB pageableByteCount:0
																	 widenedByteCount:nil
																	  layerByteCounts:@[ @(10 * NFKTestGiB), @(10 * NFKTestGiB),
																						 @(8 * NFKTestGiB) ]];
	XCTAssertNil([NFKModelResidencyPlan planForStages:@[ dense ] residency:NFKModelResidencyStreamed budget:budget
												error:&error]);
	XCTAssertEqualObjects(error.localizedDescription,
						  @"stage 0 needs 22.0 GiB streamed, two layers in flight, plus a 4.0 GiB reserve, against a 24.0 GiB working set");

	XCTAssertNil([NFKModelResidencyPlan planForStages:@[ dense ] residency:NFKModelResidencyResident budget:budget
												error:&error]);
	XCTAssertEqualObjects(error.localizedDescription,
						  @"the stages need 30.0 GiB together plus a 4.0 GiB reserve, against a 24.0 GiB working set; hold them staged or streamed");
}

- (void)testVerifyingEachStageNamesTheStageThatDoesNotLoad
{
	NSInteger budget = 24 * NFKTestGiB;
	NSArray *stages = @[ [NFKModelStageFootprint footprintWithByteCount:2 * NFKTestGiB],
						 [NFKModelStageFootprint footprintWithByteCount:30 * NFKTestGiB] ];
	NFKModelResidencyPlan *plan = [NFKModelResidencyPlan planForStages:stages residency:NFKModelResidencyStaged
																budget:budget error:NULL];
	NSError *error = nil;
	XCTAssertFalse([NFKModelResidencyPlan verifyEachStageLoads:stages plan:plan budget:budget names:@[ @"text encoder" ]
														 error:&error]);
	XCTAssertEqualObjects(error.localizedDescription,
						  @"stage 1 alone needs 30.0 GiB plus a 4.0 GiB reserve, against a 24.0 GiB working set");
	NSArray<NSString *> *names = @[ @"text encoder", @"transformer" ];
	XCTAssertFalse([NFKModelResidencyPlan verifyEachStageLoads:stages plan:plan budget:budget names:names error:&error]);
	XCTAssertEqualObjects(error.localizedDescription,
						  @"transformer alone needs 30.0 GiB plus a 4.0 GiB reserve, against a 24.0 GiB working set");
}

- (void)testAutomaticPicksTheFirstResidencyThatFits
{
	NSInteger budget = 40 * NFKTestGiB;
	NSArray *small = @[ [NFKModelStageFootprint footprintWithByteCount:10 * NFKTestGiB],
						[NFKModelStageFootprint footprintWithByteCount:5 * NFKTestGiB] ];
	XCTAssertTrue([NFKModelResidencyPlan planForStages:small residency:NFKModelResidencyAutomatic budget:budget
												 error:NULL].holdsStagesResident);
	NFKModelResidencyPlan *unknown = [NFKModelResidencyPlan planForStages:small residency:NFKModelResidencyAutomatic
																   budget:0 error:NULL];
	XCTAssertEqualObjects(unknown.residency, NFKModelResidencyStaged);

	NFKModelStageFootprint *moe = [[NFKModelStageFootprint alloc] initWithByteCount:50 * NFKTestGiB
																  pageableByteCount:40 * NFKTestGiB
																   widenedByteCount:nil layerByteCounts:nil];
	NFKModelResidencyPlan *paged = [NFKModelResidencyPlan planForStages:@[ moe ] residency:NFKModelResidencyAutomatic
																 budget:budget error:NULL];
	XCTAssertTrue(paged.pagesExperts);
	XCTAssertEqual(paged.expertCacheByteCount, (40 * NFKTestGiB - 4 * NFKTestGiB - 10 * NFKTestGiB) / 2);
	NFKModelResidencyPlan *pagedWithoutBudget = [NFKModelResidencyPlan planForStages:@[ moe ]
																		   residency:NFKModelResidencyPaged budget:0
																			   error:NULL];
	XCTAssertEqual(pagedWithoutBudget.expertCacheByteCount, 4 * NFKTestGiB);
}

- (void)testAStreamedStageHoldsTheLayersThatFit
{
	NSMutableArray<NSNumber *> *layers = [NSMutableArray array];
	for (NSInteger index = 0; index < 10; index += 1) {
		[layers addObject:@(4 * NFKTestGiB)];
	}
	NFKModelStageFootprint *dense = [[NFKModelStageFootprint alloc] initWithByteCount:50 * NFKTestGiB pageableByteCount:0
																	 widenedByteCount:nil layerByteCounts:layers];
	XCTAssertEqual(dense.streamedMinimumByteCount, 18 * NFKTestGiB);
	NFKModelResidencyPlan *plan = [NFKModelResidencyPlan planForStages:@[ dense ] residency:NFKModelResidencyAutomatic
																budget:40 * NFKTestGiB error:NULL];
	XCTAssertTrue([plan streamsStageAtIndex:0]);
	XCTAssertEqualObjects(plan.streamedStageByteCounts[@0], @(18 * NFKTestGiB));
	XCTAssertTrue(NSEqualRanges([plan heldLayerRangeOfStageAtIndex:0 footprint:dense], NSMakeRange(0, 4)));
}

- (void)testAStageWidensWhereTheWiderPrecisionFits
{
	NSArray *stages = @[ [[NFKModelStageFootprint alloc] initWithByteCount:10 * NFKTestGiB pageableByteCount:0
														  widenedByteCount:@(20 * NFKTestGiB) layerByteCounts:nil],
						 [NFKModelStageFootprint footprintWithByteCount:5 * NFKTestGiB] ];
	NFKModelResidencyPlan *roomy = [NFKModelResidencyPlan planForStages:stages residency:NFKModelResidencyResident
																 budget:40 * NFKTestGiB error:NULL];
	XCTAssertTrue([roomy widensStageAtIndex:0]);
	NFKModelResidencyPlan *tight = [NFKModelResidencyPlan planForStages:stages residency:NFKModelResidencyResident
																 budget:25 * NFKTestGiB error:NULL];
	XCTAssertFalse([tight widensStageAtIndex:0]);
	NFKModelResidencyPlan *unknown = [NFKModelResidencyPlan planForStages:stages residency:NFKModelResidencyStaged
																   budget:0 error:NULL];
	XCTAssertEqual(unknown.widenedStageIndexes.count, 0u);
}

@end
