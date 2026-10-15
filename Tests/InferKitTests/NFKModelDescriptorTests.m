//
//  NFKModelDescriptorTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

@interface NFKModelDescriptorTests : XCTestCase
@end

@implementation NFKModelDescriptorTests

// The descriptor in the shape Docs/model-catalog.md documents, every field set.
- (NSDictionary<NSString *, id> *)fullDescriptorDictionary
{
	return @{
		@"id": @"example.upscaler",
		@"displayName": @"Example Upscaler",
		@"family": @"example",
		@"summary": @"Enlarges an image four times.",
		@"engine": NFKModelEngineMLX,
		@"registryName": @"example-upscaler",
		@"providerClass": @"ExampleUpscalerProvider",
		@"tasks": @[ NFKModelTaskUpscaling, NFKModelTaskImageRestoration ],
		@"inputModalities": @[ NFKModelValueTypeImage ],
		@"outputModalities": @[ NFKModelValueTypeImage ],
		@"inputs": @[ @{ @"key": NFKInputImage, @"valueType": NFKModelValueTypeImage, @"required": @YES,
						 @"title": @"Image", @"help": @"The image to enlarge." } ],
		@"outputs": @[ @{ @"key": NFKOutputImage, @"valueType": NFKModelValueTypeImage, @"required": @YES } ],
		@"options": @[ @{ @"key": @"strength", @"kind": NFKModelOptionKindNumber, @"title": @"Strength",
							 @"help": @"How far the result moves from the source.", @"default": @0.5,
							 @"minimum": @0, @"maximum": @1, @"step": @0.05, @"stage": NFKModelOptionStageRequest,
							 @"group": @"Quality" },
						   @{ @"key": @"tile", @"kind": NFKModelOptionKindChoice,
							  @"choices": @[ @{ @"value": @"auto", @"title": @"Automatic" },
											 @{ @"value": @512, @"title": @"512 pixels" } ],
							  @"default": @"auto", @"stage": NFKModelOptionStageLoad } ],
		@"license": @{ @"id": @"Apache-2.0", @"name": @"Apache License 2.0",
					   @"url": @"https://www.apache.org/licenses/LICENSE-2.0", @"gated": @NO,
					   @"restrictsCommercialUse": @NO, @"note": @"Keep the notice." },
		@"variants": @[ @{ @"id": @"x4", @"displayName": @"×4", @"isDefault": @YES, @"registryName": @"example-x4",
						   @"source": @{ @"kind": NFKModelSourceKindHuggingFace, @"repository": @"org/example",
										 @"revision": @"0123456789abcdef", @"url": @"https://huggingface.co/org/example",
										 @"layout": NFKModelSourceLayoutDirectory, @"requiresConversion": @NO,
										 @"files": @[ @{ @"path": @"model.safetensors", @"bytes": @123,
														 @"sha256": @"ab12", @"role": NFKModelFileRoleWeights,
														 @"optional": @NO } ] },
						   @"downloadBytes": @123, @"residentBytes": @456, @"peakBytes": @600,
						   @"parameterCount": @7, @"precision": @"bfloat16", @"minimumMemoryBytes": @789 } ],
		@"requirements": @{ @"minimumOS": @{ @"macOS": @"14.0", @"iOS": @"17.0" }, @"requiresAppleSilicon": @YES,
							@"minimumMemoryBytes": @789 },
		@"residencies": @[ NFKModelResidencyResident, NFKModelResidencyStaged ],
		@"standIn": @NO,
		@"introducedVersion": @"0.4.0",
	};
}

#pragma mark Round trips

- (void)testAFullDescriptorRoundTripsThroughItsDictionary
{
	NSDictionary *dictionary = [self fullDescriptorDictionary];
	NSError *error = nil;
	NFKModelDescriptor *descriptor = [NFKModelDescriptor descriptorWithDictionary:dictionary error:&error];
	XCTAssertNotNil(descriptor, @"%@", error);
	XCTAssertEqualObjects(descriptor.dictionaryRepresentation, dictionary);
	XCTAssertTrue([NSJSONSerialization isValidJSONObject:descriptor.dictionaryRepresentation]);

	XCTAssertEqualObjects(descriptor.identifier, @"example.upscaler");
	XCTAssertEqualObjects(descriptor.providerClassName, @"ExampleUpscalerProvider");
	XCTAssertEqualObjects(descriptor.license.URL.absoluteString, @"https://www.apache.org/licenses/LICENSE-2.0");
	XCTAssertEqualObjects(descriptor.requirements.minimumOSVersions[@"macOS"], @"14.0");
	XCTAssertTrue(descriptor.requirements.requiresAppleSilicon);
	XCTAssertEqualObjects(descriptor.defaultVariant.source.files.firstObject.role, NFKModelFileRoleWeights);
	XCTAssertEqualObjects(descriptor.options.lastObject.choices.lastObject.value, @512);
}

- (void)testAPortRoundTripsThroughItsDictionary
{
	NSDictionary *dictionary = @{ @"key": NFKInputPrompt, @"valueType": NFKModelValueTypeText, @"required": @YES,
								  @"title": @"Prompt", @"help": @"What to draw." };
	NFKModelPort *port = [NFKModelPort portWithDictionary:dictionary error:NULL];
	XCTAssertEqualObjects(port.dictionaryRepresentation, dictionary);
	XCTAssertTrue(port.isRequired);
	XCTAssertEqualObjects(port, [[NFKModelPort alloc] initWithKey:NFKInputPrompt valueType:nil required:YES
															title:@"Prompt" help:@"What to draw." extensions:nil]);
}

- (void)testAChoiceRoundTripsThroughItsDictionary
{
	NSDictionary *dictionary = @{ @"value": @"fast", @"title": @"Fast" };
	NFKModelOptionChoice *choice = [NFKModelOptionChoice choiceWithDictionary:dictionary error:NULL];
	XCTAssertEqualObjects(choice.dictionaryRepresentation, dictionary);
	XCTAssertEqualObjects([NFKModelOptionChoice choiceWithValue:@3 title:nil].title, @"3");
}

- (void)testAnOptionRoundTripsThroughItsDictionary
{
	NSDictionary *dictionary = [self fullDescriptorDictionary][@"options"][0];
	NFKModelOption *option = [NFKModelOption optionWithDictionary:dictionary error:NULL];
	XCTAssertEqualObjects(option.dictionaryRepresentation, dictionary);
	XCTAssertEqualObjects(option.defaultValue, @0.5);
	XCTAssertEqualObjects(option.step, @0.05);
	XCTAssertEqualObjects(option.group, @"Quality");
}

- (void)testALicenseRoundTripsThroughItsDictionary
{
	NSDictionary *dictionary = @{ @"id": @"CC-BY-NC-4.0", @"name": @"Creative Commons BY-NC 4.0", @"gated": @YES,
								  @"restrictsCommercialUse": @YES };
	NFKModelLicense *license = [NFKModelLicense licenseWithDictionary:dictionary error:NULL];
	XCTAssertEqualObjects(license.dictionaryRepresentation, dictionary);
	XCTAssertTrue(license.isGated);
	XCTAssertTrue(license.restrictsCommercialUse);
}

- (void)testAFileRoundTripsThroughItsDictionary
{
	NSDictionary *dictionary = @{ @"path": @"tokenizer.json", @"bytes": @2048, @"sha256": @"cd34",
								  @"role": NFKModelFileRoleTokenizer, @"optional": @YES };
	NFKModelFile *file = [NFKModelFile fileWithDictionary:dictionary error:NULL];
	XCTAssertEqualObjects(file.dictionaryRepresentation, dictionary);
	XCTAssertTrue(file.isOptional);
}

- (void)testAFileFromAnotherRepositoryRoundTripsThroughItsDictionary
{
	NSDictionary *dictionary = @{ @"path": @"bigvgan_generator.pt", @"bytes": @4096, @"role": NFKModelFileRoleWeights,
								  @"optional": @NO, @"repository": @"nvidia/bigvgan_v2_24khz_100band_256x",
								  @"revision": @"0123abcd" };
	NFKModelFile *file = [NFKModelFile fileWithDictionary:dictionary error:NULL];
	XCTAssertEqualObjects(file.dictionaryRepresentation, dictionary);
	XCTAssertEqualObjects(file.repository, @"nvidia/bigvgan_v2_24khz_100band_256x");
	XCTAssertEqualObjects(file.revision, @"0123abcd");
	XCTAssertNil(file.URL);
	XCTAssertTrue(file.overridesSource);
	XCTAssertFalse([NFKModelFile fileWithPath:@"model.safetensors" role:NFKModelFileRoleWeights].overridesSource);
}

- (void)testAFileFromAURLRoundTripsThroughItsDictionary
{
	NSDictionary *dictionary = @{ @"path": @"weights.pth", @"role": NFKModelFileRoleWeights, @"optional": @YES,
								  @"url": @"https://example.com/releases/weights.pth" };
	NFKModelFile *file = [NFKModelFile fileWithDictionary:dictionary error:NULL];
	XCTAssertEqualObjects(file.dictionaryRepresentation, dictionary);
	XCTAssertEqualObjects(file.URL.absoluteString, @"https://example.com/releases/weights.pth");
	XCTAssertTrue(file.overridesSource);
	XCTAssertEqualObjects(file, [[NFKModelFile alloc] initWithPath:@"weights.pth" byteCount:nil sha256:nil
															  role:NFKModelFileRoleWeights optional:YES repository:nil
														  revision:nil URL:file.URL extensions:nil]);
}

- (void)testAMalformedFileOverrideIsRefused
{
	NSError *error = nil;
	XCTAssertNil([NFKModelFile fileWithDictionary:(@{ @"path": @"a.bin", @"repository": @3 }) error:&error]);
	XCTAssertEqual(error.code, NFKModelCatalogErrorMalformedDescriptor);
	XCTAssertTrue([error.localizedDescription containsString:@"file.repository"], @"%@", error);
}

- (void)testASourceRoundTripsThroughItsDictionary
{
	NSDictionary *dictionary = [self fullDescriptorDictionary][@"variants"][0][@"source"];
	NFKModelSource *source = [NFKModelSource sourceWithDictionary:dictionary error:NULL];
	XCTAssertEqualObjects(source.dictionaryRepresentation, dictionary);
	XCTAssertEqual(source.files.count, (NSUInteger)1);
}

- (void)testAVariantRoundTripsEveryByteCount
{
	NSDictionary *dictionary = [self fullDescriptorDictionary][@"variants"][0];
	NFKModelVariant *variant = [NFKModelVariant variantWithDictionary:dictionary error:NULL];
	XCTAssertEqualObjects(variant.dictionaryRepresentation, dictionary);
	XCTAssertEqualObjects(variant.downloadByteCount, @123);
	XCTAssertEqualObjects(variant.residentByteCount, @456);
	XCTAssertEqualObjects(variant.peakByteCount, @600);
	XCTAssertEqualObjects(variant.parameterCount, @7);
	XCTAssertEqualObjects(variant.minimumMemoryByteCount, @789);
}

- (void)testAVariantLicenseRoundTripsAndOverridesTheDescriptors
{
	NSDictionary *license = @{ @"id": @"CC-BY-4.0", @"name": @"Creative Commons Attribution 4.0",
							   @"gated": @NO, @"restrictsCommercialUse": @NO, @"note": @"Attribution required." };
	NSDictionary *dictionary = @{ @"id": @"en-de", @"displayName": @"en-de", @"isDefault": @YES, @"license": license };
	NSError *error = nil;
	NFKModelVariant *variant = [NFKModelVariant variantWithDictionary:dictionary error:&error];
	XCTAssertNotNil(variant, @"%@", error);
	XCTAssertEqualObjects(variant.dictionaryRepresentation, dictionary);
	XCTAssertEqualObjects(variant.license.identifier, @"CC-BY-4.0");

	NFKModelVariant *plain = [NFKModelVariant variantWithIdentifier:@"ja-en" displayName:nil];
	XCTAssertNil(plain.license);
	XCTAssertNil(plain.dictionaryRepresentation[@"license"]);
	NFKModelLicense *apache = [NFKModelLicense licenseWithIdentifier:@"Apache-2.0" name:@"Apache License 2.0" URL:nil];
	NFKModelDescriptor *descriptor = [[NFKModelDescriptor alloc]
		initWithIdentifier:@"opus-mt" displayName:nil family:@"marian" summary:nil engine:NFKModelEngineMLX
			  registryName:nil providerClassName:nil tasks:@[ NFKModelTaskTranslation ] inputModalities:nil
		  outputModalities:nil inputs:nil outputs:nil options:nil variants:@[ variant, plain ] license:apache
			  requirements:nil residencies:nil standIn:NO introducedVersion:nil extensions:nil];
	XCTAssertEqualObjects([descriptor licenseForVariant:variant].identifier, @"CC-BY-4.0");
	XCTAssertEqualObjects([descriptor licenseForVariant:plain], apache);
}

- (void)testAVariantWithAMalformedLicenseIsRefused
{
	NSError *error = nil;
	XCTAssertNil([NFKModelVariant variantWithDictionary:(@{ @"id": @"v", @"license": @{ @"gated": @YES } }) error:&error]);
	XCTAssertTrue([error.localizedDescription containsString:@"variant.license.name"], @"%@", error);
}

- (void)testRequirementsRoundTripThroughTheirDictionary
{
	NSDictionary *dictionary = [self fullDescriptorDictionary][@"requirements"];
	NFKModelRequirements *requirements = [NFKModelRequirements requirementsWithDictionary:dictionary error:NULL];
	XCTAssertEqualObjects(requirements.dictionaryRepresentation, dictionary);
	XCTAssertEqualObjects(requirements.minimumMemoryByteCount, @789);
}

#pragma mark Unknown fields

- (void)testUnknownFieldsAreKeptAtEveryLevelAndWrittenBack
{
	NSMutableDictionary *dictionary = [[self fullDescriptorDictionary] mutableCopy];
	dictionary[@"x-rating"] = @5;
	dictionary[@"inputs"] = @[ @{ @"key": NFKInputImage, @"x-accepts": @[ @"png" ] } ];
	dictionary[@"options"] = @[ @{ @"key": @"tile", @"kind": NFKModelOptionKindChoice,
									  @"availability": @{ @"platforms": @[ @"macOS" ] },
									  @"choices": @[ @{ @"value": @512, @"title": @"512", @"peakBytes": @1000 } ] } ];
	dictionary[@"license"] = @{ @"name": @"Custom", @"x-review": @"pending" };
	dictionary[@"requirements"] = @{ @"x-gpu": @"any" };
	dictionary[@"variants"] = @[ @{ @"id": @"v", @"x-tier": @"pro",
									@"source": @{ @"kind": NFKModelSourceKindURL, @"x-mirror": @"b",
												  @"files": @[ @{ @"path": @"m.bin", @"x-chunk": @1 } ] } } ];

	NSError *error = nil;
	NFKModelDescriptor *descriptor = [NFKModelDescriptor descriptorWithDictionary:dictionary error:&error];
	XCTAssertNotNil(descriptor, @"%@", error);
	XCTAssertEqualObjects(descriptor.extensions, (@{ @"x-rating": @5 }));
	XCTAssertEqualObjects(descriptor.inputs.firstObject.extensions, (@{ @"x-accepts": @[ @"png" ] }));
	XCTAssertEqualObjects(descriptor.options.firstObject.extensions[@"availability"], (@{ @"platforms": @[ @"macOS" ] }));
	XCTAssertEqualObjects(descriptor.options.firstObject.choices.firstObject.extensions, (@{ @"peakBytes": @1000 }));
	XCTAssertEqualObjects(descriptor.license.extensions, (@{ @"x-review": @"pending" }));
	XCTAssertEqualObjects(descriptor.requirements.extensions, (@{ @"x-gpu": @"any" }));
	XCTAssertEqualObjects(descriptor.variants.firstObject.extensions, (@{ @"x-tier": @"pro" }));
	XCTAssertEqualObjects(descriptor.variants.firstObject.source.extensions, (@{ @"x-mirror": @"b" }));
	XCTAssertEqualObjects(descriptor.variants.firstObject.source.files.firstObject.extensions, (@{ @"x-chunk": @1 }));

	NSDictionary *written = descriptor.dictionaryRepresentation;
	XCTAssertEqualObjects(written[@"x-rating"], @5);
	XCTAssertEqualObjects(written[@"inputs"][0][@"x-accepts"], (@[ @"png" ]));
	XCTAssertEqualObjects(written[@"options"][0][@"availability"], (@{ @"platforms": @[ @"macOS" ] }));
	XCTAssertEqualObjects(written[@"options"][0][@"choices"][0][@"peakBytes"], @1000);
	XCTAssertEqualObjects(written[@"license"][@"x-review"], @"pending");
	XCTAssertEqualObjects(written[@"requirements"][@"x-gpu"], @"any");
	XCTAssertEqualObjects(written[@"variants"][0][@"x-tier"], @"pro");
	XCTAssertEqualObjects(written[@"variants"][0][@"source"][@"x-mirror"], @"b");
	XCTAssertEqualObjects(written[@"variants"][0][@"source"][@"files"][0][@"x-chunk"], @1);
	XCTAssertEqualObjects([NFKModelDescriptor descriptorWithDictionary:written error:NULL], descriptor);
}

- (void)testAKnownFieldPassedAsAnExtensionIsDropped
{
	NFKModelPort *port = [[NFKModelPort alloc] initWithKey:NFKInputPrompt valueType:nil required:NO title:nil help:nil
												extensions:@{ @"key": @"other", @"x-note": @"kept" }];
	XCTAssertEqualObjects(port.extensions, (@{ @"x-note": @"kept" }));
	XCTAssertEqualObjects(port.dictionaryRepresentation[@"key"], NFKInputPrompt);
}

#pragma mark Defaults and derived fields

- (void)testAMinimalDescriptorTakesItsDefaults
{
	NSError *error = nil;
	NFKModelDescriptor *descriptor = [NFKModelDescriptor descriptorWithDictionary:@{ @"id": @"m", @"family": @"f",
																					@"engine": NFKModelEnginePlugin }
																			error:&error];
	XCTAssertNotNil(descriptor, @"%@", error);
	XCTAssertEqualObjects(descriptor.displayName, @"m");
	XCTAssertEqual(descriptor.tasks.count, (NSUInteger)0);
	XCTAssertEqual(descriptor.variants.count, (NSUInteger)0);
	XCTAssertNil(descriptor.defaultVariant);
	XCTAssertNil(descriptor.license);
	XCTAssertEqualObjects(descriptor.requirements, [[NFKModelRequirements alloc] init]);
	XCTAssertFalse(descriptor.isStandIn);

	NSDictionary *written = descriptor.dictionaryRepresentation;
	XCTAssertNil(written[@"summary"]);
	XCTAssertNil(written[@"license"]);
	XCTAssertNil(written[@"registryName"]);
	XCTAssertEqualObjects(written[@"requirements"], (@{ @"minimumOS": @{}, @"requiresAppleSilicon": @NO }));
	XCTAssertFalse([written.allValues containsObject:NSNull.null]);
}

- (void)testAnOptionDefaultsToTheRequestStageAndHidesEmptyChoices
{
	NFKModelOption *option = [NFKModelOption optionWithKey:NFKParameterSeed kind:NFKModelOptionKindInteger];
	XCTAssertEqualObjects(option.stage, NFKModelOptionStageRequest);
	XCTAssertEqual(option.choices.count, (NSUInteger)0);
	XCTAssertEqualObjects(option.dictionaryRepresentation,
						  (@{ @"key": NFKParameterSeed, @"kind": NFKModelOptionKindInteger, @"stage": NFKModelOptionStageRequest }));
}

- (void)testAnOptionWithoutAKindFollowsItsKey
{
	XCTAssertEqualObjects([NFKModelOption optionWithKey:NFKParameterTemperature].kind, NFKModelOptionKindNumber);
	XCTAssertEqualObjects([NFKModelOption optionWithKey:NFKParameterMaxTokens].kind, NFKModelOptionKindInteger);
	XCTAssertEqualObjects([NFKModelOption optionWithKey:NFKParameterGenerateAudio].kind, NFKModelOptionKindBoolean);
	XCTAssertEqualObjects([NFKModelOption optionWithKey:NFKParameterAspectRatio].kind, NFKModelOptionKindText);
	XCTAssertEqualObjects([NFKModelOption optionWithKey:NFKParameterStopSequences].kind, NFKModelOptionKindTextList);
	XCTAssertEqualObjects([NFKModelOption optionWithKey:NFKParameterJSONSchema].kind, NFKModelOptionKindJSON);
	XCTAssertEqualObjects([NFKModelOption optionWithKey:@"plugin.custom"].kind, NFKModelOptionKindJSON);
	XCTAssertEqualObjects([NFKModelOption optionWithDictionary:@{ @"key": NFKParameterSteps } error:NULL].kind,
						  NFKModelOptionKindInteger);
}

- (void)testAFileDefaultsToTheOtherRoleAndASourceToNoLayout
{
	NFKModelFile *file = [NFKModelFile fileWithDictionary:@{ @"path": @"notes.txt" } error:NULL];
	XCTAssertEqualObjects(file.role, NFKModelFileRoleOther);
	XCTAssertFalse(file.isOptional);
	NFKModelSource *source = [NFKModelSource sourceWithDictionary:@{ @"kind": NFKModelSourceKindSystem } error:NULL];
	XCTAssertEqualObjects(source.layout, NFKModelSourceLayoutNone);
	XCTAssertEqualObjects(source.dictionaryRepresentation,
						  (@{ @"kind": NFKModelSourceKindSystem, @"layout": NFKModelSourceLayoutNone, @"files": @[],
							  @"requiresConversion": @NO }));
}

- (void)testALicenseWithoutANameTakesItsIdentifier
{
	NFKModelLicense *license = [NFKModelLicense licenseWithDictionary:@{ @"id": @"MIT" } error:NULL];
	XCTAssertEqualObjects(license.name, @"MIT");
	XCTAssertFalse(license.isGated);
}

- (void)testModalitiesAreDerivedFromThePortsWhenNotGiven
{
	NFKModelDescriptor *descriptor =
		[[NFKModelDescriptor alloc] initWithIdentifier:@"vqa" displayName:nil family:@"f" summary:nil
												engine:NFKModelEngineMLX registryName:nil providerClassName:nil
												 tasks:@[ NFKModelTaskVisionQuestionAnswering ]
									   inputModalities:nil outputModalities:nil
												inputs:@[ [NFKModelPort portWithKey:NFKInputPrompt required:YES],
														  [NFKModelPort portWithKey:NFKInputImage required:YES],
														  [NFKModelPort portWithKey:NFKInputNegativePrompt required:NO],
														  [NFKModelPort portWithKey:@"plugin.unknown" required:NO] ]
											   outputs:@[ [NFKModelPort outputPortWithKey:NFKOutputText required:YES] ]
											options:nil variants:nil license:nil requirements:nil residencies:nil
											   standIn:NO introducedVersion:nil extensions:nil];
	XCTAssertEqualObjects(descriptor.inputModalities, (@[ NFKModelValueTypeText, NFKModelValueTypeImage ]));
	XCTAssertEqualObjects(descriptor.outputModalities, (@[ NFKModelValueTypeText ]));
	XCTAssertNil(descriptor.inputs.lastObject.valueType);

	NSDictionary *given = @{ @"id": @"m", @"family": @"f", @"engine": NFKModelEngineMLX,
							 @"inputs": @[ @{ @"key": NFKInputImage } ], @"inputModalities": @[ NFKModelValueTypeVideo ] };
	XCTAssertEqualObjects([NFKModelDescriptor descriptorWithDictionary:given error:NULL].inputModalities,
						  (@[ NFKModelValueTypeVideo ]));
}

- (void)testTheDefaultVariantIsTheMarkedOneElseTheFirst
{
	NFKModelVariant *small = [NFKModelVariant variantWithIdentifier:@"small" displayName:nil];
	NFKModelVariant *large = [[NFKModelVariant alloc] initWithIdentifier:@"large" displayName:@"Large" isDefault:YES
															registryName:nil source:nil downloadByteCount:nil
													   residentByteCount:nil peakByteCount:nil parameterCount:nil
															   precision:nil minimumMemoryByteCount:nil extensions:nil];
	NFKModelDescriptor *marked = [self descriptorWithVariants:@[ small, large ]];
	XCTAssertEqualObjects(marked.defaultVariant.identifier, @"large");
	XCTAssertEqualObjects([marked variantWithIdentifier:@"small"], small);
	XCTAssertNil([marked variantWithIdentifier:@"medium"]);
	XCTAssertEqualObjects([self descriptorWithVariants:@[ small ]].defaultVariant.identifier, @"small");
	XCTAssertEqualObjects(small.displayName, @"small");
}

- (NFKModelDescriptor *)descriptorWithVariants:(NSArray<NFKModelVariant *> *)variants
{
	return [[NFKModelDescriptor alloc] initWithIdentifier:@"m" displayName:nil family:@"f" summary:nil
												   engine:NFKModelEngineMLX registryName:nil providerClassName:nil
													tasks:nil inputModalities:nil outputModalities:nil inputs:nil
												  outputs:nil options:nil variants:variants license:nil
											 requirements:nil residencies:nil standIn:NO introducedVersion:nil
											   extensions:nil];
}

#pragma mark Value types

- (void)testAPortWithoutAValueTypeTakesTheCoreKeysType
{
	XCTAssertEqualObjects([NFKModelPort portWithKey:NFKInputPrompt required:YES].valueType, NFKModelValueTypeText);
	XCTAssertEqualObjects([NFKModelPort portWithKey:NFKInputImage required:YES].valueType, NFKModelValueTypeImage);
	XCTAssertEqualObjects([NFKModelPort portWithDictionary:@{ @"key": NFKInputAudio } error:NULL].valueType,
						  NFKModelValueTypeAudio);
	XCTAssertEqualObjects([NFKModelPort outputPortWithKey:NFKOutputPose required:NO].valueType, NFKModelValueTypeKeypoints);
	XCTAssertEqualObjects([NFKModelPort portWithKey:@"custom" required:NO].dictionaryRepresentation,
						  (@{ @"key": @"custom", @"required": @NO }));
}

- (void)testASharedKeyStringResolvesByItsSide
{
	XCTAssertEqualObjects(NFKModelValueTypeForKey(@"citations"), NFKModelValueTypeBoolean);
	XCTAssertEqualObjects(NFKModelValueTypeForOutputKey(@"citations"), NFKModelValueTypeJSON);
	XCTAssertEqualObjects([NFKModelPort outputPortWithKey:NFKOutputCitations required:NO].valueType, NFKModelValueTypeJSON);

	NSDictionary *dictionary = @{ @"id": @"m", @"family": @"f", @"engine": NFKModelEngineMLX,
								  @"outputs": @[ @{ @"key": NFKOutputCitations } ] };
	NFKModelDescriptor *descriptor = [NFKModelDescriptor descriptorWithDictionary:dictionary error:NULL];
	XCTAssertEqualObjects(descriptor.outputs.firstObject.valueType, NFKModelValueTypeJSON);
}

- (void)testAnUnknownKeyHasNoValueType
{
	XCTAssertNil(NFKModelValueTypeForKey(@"plugin.unknown"));
	XCTAssertNil(NFKModelValueTypeForOutputKey(@"plugin.unknown"));
}

- (void)testTheValueTypeTableCoversEveryCoreKey
{
	NSArray<NSString *> *inputs = @[
		NFKInputPrompt, NFKInputNegativePrompt, NFKInputImage, NFKInputMask, NFKInputImages, NFKInputDocument,
		NFKInputDocuments, NFKInputControl, NFKInputVideo, NFKInputLastFrame, NFKInputVoiceReference, NFKInputSuffix,
		NFKInputAudio, NFKInputAudios, NFKInputMessages, NFKInputLyrics, NFKInputState, NFKInputQuestions,
	];
	NSArray<NSString *> *parameters = @[
		NFKParameterSeed, NFKParameterSteps, NFKParameterGuidanceScale, NFKParameterStrength, NFKParameterWidth,
		NFKParameterHeight, NFKParameterTools, NFKParameterMCPServers, NFKParameterJSONSchema,
		NFKParameterOutputFormat, NFKParameterChoices, NFKParameterVideoFrameCount, NFKParameterAudioOutput,
		NFKParameterSampleCount, NFKParameterFrameCount, NFKParameterFramesPerSecond, NFKParameterDurationSeconds,
		NFKParameterAspectRatio, NFKParameterResolution, NFKParameterGenerateAudio, NFKParameterVideoOperation,
		NFKParameterSourceVideoIdentifier, NFKParameterSpeakerDiarization, NFKParameterWordTimestamps,
		NFKParameterVocabulary, NFKParameterCitations, NFKParameterPreviousResponseIdentifier,
		NFKParameterConversationKey, NFKParameterMotionScale, NFKParameterSampleRate, NFKParameterChannelCount,
		NFKParameterSourceLanguage, NFKParameterTargetLanguage, NFKParameterTemperature, NFKParameterTopP,
		NFKParameterTopK, NFKParameterMaxTokens, NFKParameterRepetitionPenalty, NFKParameterStopSequences,
		NFKParameterReasoningEffort,
	];
	NSArray<NSString *> *outputs = @[
		NFKOutputImage, NFKOutputImages, NFKOutputVideo, NFKOutputVideos, NFKOutputWords, NFKOutputCitations,
		NFKOutputServerToolResults, NFKOutputResponseIdentifier, NFKOutputAudio, NFKOutputMask, NFKOutputText,
		NFKOutputStructured, NFKOutputEmbedding, NFKOutputToolCalls, NFKOutputAssistantMessage, NFKOutputMessages,
		NFKOutputToolInvocations, NFKOutputToolTurnLimitReached, NFKOutputDetections, NFKOutputPose,
		NFKOutputClassifications, NFKOutputSegments, NFKOutputMIDI, NFKOutputBeats, NFKOutputTempo,
		NFKOutputAnswers, NFKOutputReasoning, NFKOutputUsage,
	];
	XCTAssertEqual(inputs.count + parameters.count + outputs.count, (NSUInteger)86);
	for (NSString *key in [inputs arrayByAddingObjectsFromArray:parameters]) {
		XCTAssertNotNil(NFKModelValueTypeForKey(key), @"%@ has no value type", key);
	}
	for (NSString *key in outputs) {
		XCTAssertNotNil(NFKModelValueTypeForOutputKey(key), @"%@ has no value type", key);
	}
	XCTAssertEqualObjects(NFKModelValueTypeForKey(NFKParameterSeed), NFKModelValueTypeInteger);
	XCTAssertEqualObjects(NFKModelValueTypeForKey(NFKParameterTemperature), NFKModelValueTypeNumber);
	XCTAssertEqualObjects(NFKModelValueTypeForKey(NFKInputMessages), NFKModelValueTypeMessages);
	XCTAssertEqualObjects(NFKModelValueTypeForOutputKey(NFKOutputToolCalls), NFKModelValueTypeToolCalls);
	XCTAssertEqualObjects(NFKModelValueTypeForOutputKey(NFKOutputMIDI), NFKModelValueTypeMIDI);
}

- (void)testTasksThatACapabilityNamesShareItsString
{
	XCTAssertEqualObjects(NFKModelTaskTextGeneration, NFKCapabilityTextGeneration);
	XCTAssertEqualObjects(NFKModelTaskTranscription, NFKCapabilityTranscription);
	XCTAssertEqualObjects(NFKModelTaskTranslation, NFKCapabilityTranslation);
	XCTAssertEqualObjects(NFKModelTaskTextRecognition, NFKCapabilityTextRecognition);
	XCTAssertEqualObjects(NFKModelTaskSegmentation, NFKCapabilitySegmentation);
	XCTAssertEqualObjects(NFKModelTaskPoseEstimation, NFKCapabilityPose);
	XCTAssertEqualObjects(NFKModelTaskFaceDetection, NFKCapabilityFaceDetection);
	XCTAssertEqualObjects(NFKModelTaskImageEmbedding, NFKCapabilityImageEmbedding);
	XCTAssertEqualObjects(NFKModelTaskUpscaling, NFKCapabilityUpscaling);
	XCTAssertEqualObjects(NFKModelTaskOpticalFlow, NFKCapabilityOpticalFlow);
	XCTAssertEqualObjects(NFKModelTaskControlNet, NFKCapabilityControlNet);
}

- (void)testTheDecisionTaskIsTheDecisionString
{
	XCTAssertEqualObjects(NFKModelTaskDecision, @"decision");
	NSDictionary *dictionary = @{ @"id": @"m", @"family": @"f", @"engine": NFKModelEngineMLX,
								  @"tasks": @[ NFKModelTaskDecision, NFKModelTaskClassification ] };
	NFKModelDescriptor *descriptor = [NFKModelDescriptor descriptorWithDictionary:dictionary error:NULL];
	XCTAssertEqualObjects(descriptor.tasks.firstObject, NFKModelTaskDecision);
}

#pragma mark Malformed input

- (void)testAMissingRequiredFieldFailsWithItsPath
{
	NSError *error = nil;
	NSDictionary *withoutIdentifier = @{ @"family": @"f", @"engine": @"mlx" };
	XCTAssertNil([NFKModelDescriptor descriptorWithDictionary:withoutIdentifier error:&error]);
	XCTAssertEqualObjects(error.domain, NFKModelCatalogErrorDomain);
	XCTAssertEqual(error.code, NFKModelCatalogErrorMalformedDescriptor);
	XCTAssertTrue([error.localizedDescription containsString:@"model.id"], @"%@", error.localizedDescription);

	error = nil;
	XCTAssertNil([NFKModelPort portWithDictionary:@{ @"required": @YES } error:&error]);
	XCTAssertEqual(error.code, NFKModelCatalogErrorMalformedDescriptor);
	XCTAssertNil([NFKModelSource sourceWithDictionary:@{} error:NULL]);
	XCTAssertNil([NFKModelFile fileWithDictionary:@{} error:NULL]);
	XCTAssertNil([NFKModelVariant variantWithDictionary:@{} error:NULL]);
	XCTAssertNil([NFKModelLicense licenseWithDictionary:@{} error:NULL]);
	XCTAssertNil([NFKModelOptionChoice choiceWithDictionary:@{ @"title": @"No value" } error:NULL]);
}

- (void)testAWrongTypeDeepInsideFailsWithItsPath
{
	NSMutableDictionary *dictionary = [[self fullDescriptorDictionary] mutableCopy];
	dictionary[@"variants"] = @[ @{ @"id": @"v", @"source": @{ @"kind": @"url",
																@"files": @[ @{ @"path": @"a", @"bytes": @"many" } ] } } ];
	NSError *error = nil;
	XCTAssertNil([NFKModelDescriptor descriptorWithDictionary:dictionary error:&error]);
	XCTAssertEqual(error.code, NFKModelCatalogErrorMalformedDescriptor);
	XCTAssertTrue([error.localizedDescription containsString:@"model.variants[0].source.files[0].bytes"],
				  @"%@", error.localizedDescription);
}

- (void)testMalformedValuesAreRefused
{
	NSDictionary *base = @{ @"id": @"m", @"family": @"f", @"engine": @"mlx" };
	NSArray<NSDictionary *> *changes = @[
		@{ @"tasks": @"upscaling" },
		@{ @"tasks": @[ @1 ] },
		@{ @"standIn": @"no" },
		@{ @"inputs": @{ @"key": @"image" } },
		@{ @"inputs": @[ @"image" ] },
		@{ @"license": @"MIT" },
		@{ @"requirements": @{ @"minimumOS": @{ @"macOS": @14 } } },
		@{ @"variants": @[ @{ @"id": @"v", @"residentBytes": @YES } ] },
		@{ @"options": @[ @{ @"key": @"k", @"default": NSNull.null } ] },
		@{ @"options": @[ @{ @"key": @"k", @"choices": @[ @{ @"value": @[ @1 ] } ] } ] },
		@{ @"license": @{ @"name": @"n", @"url": @42 } },
	];
	for (NSDictionary *change in changes) {
		NSMutableDictionary *dictionary = [base mutableCopy];
		[dictionary addEntriesFromDictionary:change];
		NSError *error = nil;
		XCTAssertNil([NFKModelDescriptor descriptorWithDictionary:dictionary error:&error], @"%@ was accepted", change);
		XCTAssertEqual(error.code, NFKModelCatalogErrorMalformedDescriptor, @"%@", change);
	}
}

#pragma mark Identity

- (void)testDescriptorsAreEqualWhenTheirDictionariesAre
{
	NFKModelDescriptor *first = [NFKModelDescriptor descriptorWithDictionary:[self fullDescriptorDictionary] error:NULL];
	NFKModelDescriptor *second = [NFKModelDescriptor descriptorWithDictionary:[self fullDescriptorDictionary] error:NULL];
	XCTAssertEqualObjects(first, second);
	XCTAssertEqual(first.hash, second.hash);
	XCTAssertEqual([first copy], first);
	XCTAssertNotEqualObjects(first, [self descriptorWithVariants:nil]);
}

@end
