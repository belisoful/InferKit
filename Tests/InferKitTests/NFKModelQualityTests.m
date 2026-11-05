//
//  NFKModelQualityTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

@interface NFKModelQualityTests : XCTestCase
@end

@implementation NFKModelQualityTests

- (NSDictionary *)scoreForTask:(NSString *)task benchmark:(NSString *)benchmark snapshot:(NSString *)snapshot
						 value:(double)value
{
	return @{ @"task": task, @"benchmark": benchmark, @"snapshot": snapshot, @"metric": @"accuracy",
			  @"value": @(value), @"source": @"https://example.com/report" };
}

- (NFKModelDescriptor *)descriptor:(NSString *)identifier scores:(NSArray<NSDictionary *> *)scores
{
	NSDictionary *model = @{ @"id": identifier, @"family": @"f", @"engine": NFKModelEngineMLX,
							 @"tasks": @[ NFKModelTaskTextGeneration, NFKModelTaskCodeGeneration ],
							 @"qualityScores": scores };
	NSError *error = nil;
	NFKModelDescriptor *descriptor = [NFKModelDescriptor descriptorWithDictionary:model error:&error];
	XCTAssertNotNil(descriptor, @"%@", error);
	return descriptor;
}

- (void)testAScoreRoundTripsThroughTheDictionary
{
	NSDictionary *dictionary = @{ @"task": NFKModelTaskTranscription, @"benchmark": NFKModelBenchmarkOpenASRLeaderboard,
								  @"snapshot": @"2026-10", @"metric": @"WER", @"value": @7.44, @"higherIsBetter": @NO,
								  @"variant": @"fp16", @"source": @"https://huggingface.co/spaces/hf-audio/open_asr_leaderboard",
								  @"retrieved": @"2026-11-05", @"note": @"mean of eight test sets", @"extra": @1 };
	NSError *error = nil;
	NFKModelQualityScore *score = [NFKModelQualityScore scoreWithDictionary:dictionary error:&error];
	XCTAssertNotNil(score, @"%@", error);
	XCTAssertEqualObjects(score.benchmarkIdentifier, @"open-asr-leaderboard@2026-10");
	XCTAssertEqual(score.value, 7.44);
	XCTAssertFalse(score.higherIsBetter);
	XCTAssertEqualObjects(score.variantIdentifier, @"fp16");
	XCTAssertEqualObjects(score.retrievedDate, @"2026-11-05");
	XCTAssertEqualObjects(score.extensions, @{ @"extra": @1 });
	XCTAssertEqualObjects(score.dictionaryRepresentation, dictionary);
}

- (void)testAnAbsentHigherIsBetterIsYes
{
	NFKModelQualityScore *score = [NFKModelQualityScore scoreWithDictionary:[self scoreForTask:NFKModelTaskTextGeneration
																					 benchmark:NFKModelBenchmarkMMLUPro
																					  snapshot:@"1" value:50]
																	  error:NULL];
	XCTAssertTrue(score.higherIsBetter);
}

- (void)testAScoreWithoutValueOrSourceIsRefused
{
	for (NSString *field in @[ @"value", @"source", @"snapshot", @"metric" ]) {
		NSMutableDictionary *dictionary = [[self scoreForTask:NFKModelTaskTextGeneration benchmark:NFKModelBenchmarkMMLUPro
													 snapshot:@"1" value:50] mutableCopy];
		[dictionary removeObjectForKey:field];
		NSError *error = nil;
		XCTAssertNil([NFKModelQualityScore scoreWithDictionary:dictionary error:&error], @"%@", field);
		XCTAssertEqual(error.code, NFKModelCatalogErrorMalformedDescriptor, @"%@", field);
		XCTAssertTrue([error.localizedDescription containsString:field], @"%@", error.localizedDescription);
	}
}

- (void)testScoresOfDifferentSnapshotsDoNotCompare
{
	NFKModelQualityScore *older = [NFKModelQualityScore scoreWithDictionary:[self scoreForTask:NFKModelTaskCodeGeneration
																					 benchmark:NFKModelBenchmarkLiveCodeBench
																					  snapshot:@"v5" value:60]
																	  error:NULL];
	NFKModelQualityScore *newer = [NFKModelQualityScore scoreWithDictionary:[self scoreForTask:NFKModelTaskCodeGeneration
																					 benchmark:NFKModelBenchmarkLiveCodeBench
																					  snapshot:@"v6" value:40]
																	  error:NULL];
	XCTAssertFalse([older isBetterThanScore:newer]);
	XCTAssertFalse([newer isBetterThanScore:older]);
}

- (void)testDescriptorScoresRoundTripAndBelongToTheirVariant
{
	NSDictionary *base = [self scoreForTask:NFKModelTaskTextGeneration benchmark:NFKModelBenchmarkMMLUPro snapshot:@"1"
									  value:69.6];
	NSMutableDictionary *quantized = [base mutableCopy];
	quantized[@"variant"] = @"4bit";
	quantized[@"value"] = @66.1;
	NSDictionary *model = @{ @"id": @"m", @"family": @"f", @"engine": NFKModelEngineMLX,
							 @"tasks": @[ NFKModelTaskTextGeneration ],
							 @"variants": @[ @{ @"id": @"bf16", @"default": @YES }, @{ @"id": @"4bit" } ],
							 @"qualityScores": @[ base, quantized ] };
	NSError *error = nil;
	NFKModelDescriptor *descriptor = [NFKModelDescriptor descriptorWithDictionary:model error:&error];
	XCTAssertNotNil(descriptor, @"%@", error);
	XCTAssertEqual(descriptor.qualityScores.count, 2u);
	XCTAssertNil(descriptor.extensions[@"qualityScores"]);
	XCTAssertEqualObjects([[descriptor qualityScoresForVariant:nil] valueForKey:@"value"], @[ @69.6 ]);
	XCTAssertEqualObjects([[descriptor qualityScoresForVariant:@"bf16"] valueForKey:@"value"], @[ @69.6 ]);
	XCTAssertEqualObjects([[descriptor qualityScoresForVariant:@"4bit"] valueForKey:@"value"], @[ @66.1 ]);
	NFKModelDescriptor *copied = [NFKModelDescriptor descriptorWithDictionary:descriptor.dictionaryRepresentation
																		 error:&error];
	XCTAssertEqualObjects(copied, descriptor, @"%@", error);
}

- (void)testADescriptorWithoutScoresWritesNone
{
	NSDictionary *model = @{ @"id": @"m", @"family": @"f", @"engine": NFKModelEngineMLX };
	NFKModelDescriptor *descriptor = [NFKModelDescriptor descriptorWithDictionary:model error:NULL];
	XCTAssertEqualObjects(descriptor.qualityScores, @[]);
	XCTAssertNil(descriptor.dictionaryRepresentation[@"qualityScores"]);
}

- (void)testAScoreOutsideTheDescriptorIsRefused
{
	NSDictionary *unlistedTask = [self scoreForTask:NFKModelTaskTranslation benchmark:NFKModelBenchmarkFLORES200
										   snapshot:@"devtest" value:50];
	NSMutableDictionary *unknownVariant = [[self scoreForTask:NFKModelTaskTextGeneration benchmark:NFKModelBenchmarkMMLUPro
													 snapshot:@"1" value:50] mutableCopy];
	unknownVariant[@"variant"] = @"8bit";
	NSDictionary *repeated = [self scoreForTask:NFKModelTaskTextGeneration benchmark:NFKModelBenchmarkMMLUPro
									   snapshot:@"1" value:50];
	NSArray<NSArray *> *cases = @[ @[ unlistedTask ], @[ unknownVariant ], @[ repeated, repeated ] ];
	for (NSArray *scores in cases) {
		NSDictionary *model = @{ @"id": @"m", @"family": @"f", @"engine": NFKModelEngineMLX,
								 @"tasks": @[ NFKModelTaskTextGeneration ], @"qualityScores": scores };
		NSError *error = nil;
		XCTAssertNil([NFKModelDescriptor descriptorWithDictionary:model error:&error], @"%@", scores);
		XCTAssertEqual(error.code, NFKModelCatalogErrorMalformedDescriptor);
		XCTAssertTrue([error.localizedDescription containsString:@"qualityScores"], @"%@", error.localizedDescription);
	}
}

- (void)testEachRankedTaskHasItsReferenceBenchmark
{
	XCTAssertEqualObjects([NFKModelCatalog referenceBenchmarkForTask:NFKModelTaskTextGeneration], NFKModelBenchmarkMMLUPro);
	XCTAssertEqualObjects([NFKModelCatalog referenceBenchmarkForTask:NFKModelTaskCodeGeneration],
						  NFKModelBenchmarkLiveCodeBench);
	XCTAssertEqualObjects([NFKModelCatalog referenceBenchmarkForTask:NFKModelTaskTranscription],
						  NFKModelBenchmarkOpenASRLeaderboard);
	XCTAssertEqualObjects([NFKModelCatalog referenceBenchmarkForTask:NFKModelTaskTranslation], NFKModelBenchmarkFLORES200);
	XCTAssertEqualObjects([NFKModelCatalog referenceBenchmarkForTask:NFKModelTaskTextToImage], NFKModelBenchmarkGenEval);
	XCTAssertEqualObjects([NFKModelCatalog referenceBenchmarkForTask:NFKModelTaskSpeechSynthesis], NFKModelBenchmarkTTSArena);
	XCTAssertNil([NFKModelCatalog referenceBenchmarkForTask:NFKModelTaskMatting]);
	XCTAssertEqualObjects(NFKModelTaskCodeGeneration, @"code-generation");
}

- (void)testARankingOrdersBestFirstAndEqualScoresShareARank
{
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	[catalog addDescriptors:@[
		[self descriptor:@"c" scores:@[ [self scoreForTask:NFKModelTaskTextGeneration benchmark:NFKModelBenchmarkMMLUPro
												  snapshot:@"1" value:70] ]],
		[self descriptor:@"a" scores:@[ [self scoreForTask:NFKModelTaskTextGeneration benchmark:NFKModelBenchmarkMMLUPro
												  snapshot:@"1" value:55] ]],
		[self descriptor:@"b" scores:@[ [self scoreForTask:NFKModelTaskTextGeneration benchmark:NFKModelBenchmarkMMLUPro
												  snapshot:@"1" value:70] ]],
		[self descriptor:@"unscored" scores:@[]],
	]];
	NSArray<NFKModelQualityRanking *> *rankings = [catalog qualityRankingsForTask:NFKModelTaskTextGeneration];
	XCTAssertEqualObjects([rankings valueForKey:@"modelIdentifier"], (@[ @"b", @"c", @"a" ]));
	XCTAssertEqualObjects([rankings valueForKey:@"rank"], (@[ @1, @1, @3 ]));
	XCTAssertEqual(rankings.firstObject.rankedCount, 3);
	XCTAssertEqualObjects(rankings.firstObject.score.sourceURL.absoluteString, @"https://example.com/report");
	XCTAssertEqual([catalog qualityRankingForModelIdentifier:@"a" task:NFKModelTaskTextGeneration].rank, 3);
	XCTAssertNil([catalog qualityRankingForModelIdentifier:@"unscored" task:NFKModelTaskTextGeneration]);
	XCTAssertEqualObjects([catalog qualityRankingsForTask:NFKModelTaskCodeGeneration], @[]);
	XCTAssertEqualObjects([catalog qualityRankingsForTask:NFKModelTaskMatting], @[]);
}

- (void)testALowerIsBetterBenchmarkRanksTheSmallestFirst
{
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	for (NSArray *entry in @[ @[ @"large", @6.5 ], @[ @"small", @9.1 ] ]) {
		NSMutableDictionary *score = [[self scoreForTask:NFKModelTaskTranscription
											   benchmark:NFKModelBenchmarkOpenASRLeaderboard snapshot:@"2026-10"
												   value:[entry[1] doubleValue]] mutableCopy];
		score[@"higherIsBetter"] = @NO;
		NSDictionary *model = @{ @"id": entry[0], @"family": @"whisper", @"engine": NFKModelEngineMLX,
								 @"tasks": @[ NFKModelTaskTranscription ], @"qualityScores": @[ score ] };
		[catalog addDescriptor:[NFKModelDescriptor descriptorWithDictionary:model error:NULL]];
	}
	NSArray<NFKModelQualityRanking *> *rankings = [catalog qualityRankingsForTask:NFKModelTaskTranscription];
	XCTAssertEqualObjects([rankings valueForKey:@"modelIdentifier"], (@[ @"large", @"small" ]));
}

- (void)testARankingComparesOnTheSnapshotMostDescriptorsReport
{
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	[catalog addDescriptors:@[
		[self descriptor:@"old" scores:@[ [self scoreForTask:NFKModelTaskCodeGeneration
												   benchmark:NFKModelBenchmarkLiveCodeBench snapshot:@"v5" value:90] ]],
		[self descriptor:@"x" scores:@[ [self scoreForTask:NFKModelTaskCodeGeneration
												 benchmark:NFKModelBenchmarkLiveCodeBench snapshot:@"v6" value:40] ]],
		[self descriptor:@"y" scores:@[ [self scoreForTask:NFKModelTaskCodeGeneration
												 benchmark:NFKModelBenchmarkLiveCodeBench snapshot:@"v6" value:50] ]],
	]];
	NSArray<NFKModelQualityRanking *> *rankings = [catalog qualityRankingsForTask:NFKModelTaskCodeGeneration];
	XCTAssertEqualObjects([rankings valueForKey:@"modelIdentifier"], (@[ @"y", @"x" ]));
	[catalog addDescriptor:[self descriptor:@"z" scores:@[ [self scoreForTask:NFKModelTaskCodeGeneration
																	 benchmark:NFKModelBenchmarkLiveCodeBench
																	  snapshot:@"v5" value:10] ]]];
	rankings = [catalog qualityRankingsForTask:NFKModelTaskCodeGeneration];
	XCTAssertEqualObjects([rankings valueForKey:@"modelIdentifier"], (@[ @"y", @"x" ]), @"a tie takes the later snapshot");
}

- (void)testARankingUsesTheDefaultVariantAndSkipsStandIns
{
	NSMutableDictionary *quantized = [[self scoreForTask:NFKModelTaskTextGeneration benchmark:NFKModelBenchmarkMMLUPro
												snapshot:@"1" value:99] mutableCopy];
	quantized[@"variant"] = @"4bit";
	NSDictionary *variantModel = @{ @"id": @"variant-only", @"family": @"f", @"engine": NFKModelEngineMLX,
									@"tasks": @[ NFKModelTaskTextGeneration ],
									@"variants": @[ @{ @"id": @"bf16" }, @{ @"id": @"4bit" } ],
									@"qualityScores": @[ quantized ] };
	NSDictionary *standIn = @{ @"id": @"stand-in", @"family": @"f", @"engine": NFKModelEngineMLX, @"standIn": @YES,
							   @"tasks": @[ NFKModelTaskTextGeneration ],
							   @"qualityScores": @[ [self scoreForTask:NFKModelTaskTextGeneration
															 benchmark:NFKModelBenchmarkMMLUPro snapshot:@"1" value:98] ] };
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	[catalog addDescriptors:@[
		[NFKModelDescriptor descriptorWithDictionary:variantModel error:NULL],
		[NFKModelDescriptor descriptorWithDictionary:standIn error:NULL],
		[self descriptor:@"ranked" scores:@[ [self scoreForTask:NFKModelTaskTextGeneration benchmark:NFKModelBenchmarkMMLUPro
													   snapshot:@"1" value:40] ]],
	]];
	NSArray<NFKModelQualityRanking *> *rankings = [catalog qualityRankingsForTask:NFKModelTaskTextGeneration];
	XCTAssertEqualObjects([rankings valueForKey:@"modelIdentifier"], (@[ @"ranked" ]));
}

@end
