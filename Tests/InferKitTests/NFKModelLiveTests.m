//
//  NFKModelLiveTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

@interface NFKModelLiveTests : XCTestCase
@end

@implementation NFKModelLiveTests

- (NSDictionary *)timingAtWidth:(NSInteger)width height:(NSInteger)height milliseconds:(double)milliseconds
{
	return @{ @"chip": @"Apple M3 Max", @"width": @(width), @"height": @(height), @"milliseconds": @(milliseconds),
			  @"measured": @"2026-11-06" };
}

- (NSDictionary *)modelNamed:(NSString *)identifier live:(nullable NSDictionary *)live
{
	NSMutableDictionary *model = [@{ @"id": identifier, @"family": @"f", @"engine": NFKModelEngineMLX,
									 @"tasks": @[ NFKModelTaskMatting ],
									 @"inputs": @[ @{ @"key": NFKInputImage, @"required": @YES } ],
									 @"outputs": @[ @{ @"key": NFKOutputMask }, @{ @"key": NFKOutputImage } ],
									 @"variants": @[ @{ @"id": @"fp16", @"isDefault": @YES }, @{ @"id": @"q4" } ] } mutableCopy];
	model[@"live"] = live;
	return model;
}

- (nullable NFKModelDescriptor *)descriptorNamed:(NSString *)identifier live:(nullable NSDictionary *)live error:(NSError **)error
{
	return [NFKModelDescriptor descriptorWithDictionary:[self modelNamed:identifier live:live] error:error];
}

- (NSDictionary *)videoLiveWithTimings:(NSArray<NSDictionary *> *)timings
{
	return @{ @"unit": NFKModelLiveUnitVideoFrame, @"inputs": @[ NFKInputImage ], @"outputs": @[ NFKOutputMask ],
			  @"timings": timings };
}

#pragma mark Timings

- (void)testATimingRoundTripsThroughTheDictionary
{
	NSDictionary *dictionary = @{ @"variant": @"q4", @"chip": @"Apple M3 Max", @"width": @1920, @"height": @1080,
								  @"milliseconds": @12.5, @"measured": @"2026-11-06", @"note": @"GPU, float16", @"extra": @1 };
	NSError *error = nil;
	NFKModelLiveTiming *timing = [NFKModelLiveTiming timingWithDictionary:dictionary error:&error];
	XCTAssertNotNil(timing, @"%@", error);
	XCTAssertEqualObjects(timing.variantIdentifier, @"q4");
	XCTAssertEqual(timing.width, 1920);
	XCTAssertEqual(timing.height, 1080);
	XCTAssertEqual(timing.millisecondsPerUnit, 12.5);
	XCTAssertEqualObjects(timing.extensions, @{ @"extra": @1 });
	XCTAssertEqualObjects(timing.dictionaryRepresentation, dictionary);
	XCTAssertEqualObjects([NFKModelLiveTiming timingWithDictionary:dictionary error:NULL], timing);
}

- (void)testATimingWithoutAPositiveTimeIsRefused
{
	NSError *error = nil;
	NSMutableDictionary *dictionary = [[self timingAtWidth:1280 height:720 milliseconds:0] mutableCopy];
	XCTAssertNil([NFKModelLiveTiming timingWithDictionary:dictionary error:&error]);
	XCTAssertTrue([error.localizedDescription containsString:@"milliseconds"], @"%@", error);
	[dictionary removeObjectForKey:@"milliseconds"];
	XCTAssertNil([NFKModelLiveTiming timingWithDictionary:dictionary error:&error]);
	XCTAssertTrue([error.localizedDescription containsString:@"milliseconds"], @"%@", error);
	dictionary = [[self timingAtWidth:1280 height:720 milliseconds:4] mutableCopy];
	[dictionary removeObjectForKey:@"chip"];
	XCTAssertNil([NFKModelLiveTiming timingWithDictionary:dictionary error:&error]);
	XCTAssertTrue([error.localizedDescription containsString:@"chip"], @"%@", error);
}

#pragma mark Profiles

- (void)testAVideoAndAnAudioProfileRoundTripThroughTheDictionary
{
	NSDictionary *video = @{ @"unit": NFKModelLiveUnitVideoFrame, @"inputs": @[ NFKInputImage ],
							 @"outputs": @[ NFKOutputMask, NFKOutputImage ], @"stateful": @YES,
							 @"timings": @[ [self timingAtWidth:1280 height:720 milliseconds:8] ], @"extra": @"kept" };
	NSDictionary *audio = @{ @"unit": NFKModelLiveUnitAudioChunk, @"inputs": @[ NFKInputAudio ], @"outputs": @[ NFKOutputAudio ],
							 @"stateful": @NO, @"chunkDuration": @0.01, @"sampleRate": @48000, @"channelCount": @1, @"lookahead": @20 };
	NSError *error = nil;
	NFKModelLiveProfile *profile = [NFKModelLiveProfile profileWithDictionary:video error:&error];
	XCTAssertNotNil(profile, @"%@", error);
	XCTAssertTrue(profile.isStateful);
	XCTAssertEqual(profile.timings.count, 1u);
	XCTAssertEqualObjects(profile.dictionaryRepresentation, video);
	profile = [NFKModelLiveProfile profileWithDictionary:audio error:&error];
	XCTAssertNotNil(profile, @"%@", error);
	XCTAssertEqual(profile.chunkDuration, 0.01);
	XCTAssertEqual(profile.sampleRate, 48000);
	XCTAssertEqual(profile.channelCount, 1);
	XCTAssertEqual(profile.lookaheadMilliseconds, 20);
	XCTAssertEqualObjects(profile.dictionaryRepresentation, audio);
	NFKModelLiveProfile *built = [NFKModelLiveProfile audioChunkProfileWithInputKeys:@[ NFKInputAudio ] outputKeys:@[ NFKOutputAudio ]
																			stateful:NO chunkDuration:0.01 sampleRate:48000
																		channelCount:1 lookaheadMilliseconds:20 timings:nil];
	XCTAssertEqualObjects(built, profile);
}

- (void)testAMalformedProfileIsRefusedNamingItsField
{
	NSDictionary<NSString *, NSDictionary *> *cases = @{
		@"unit": @{ @"unit": @"textLine", @"inputs": @[ NFKInputImage ], @"outputs": @[ NFKOutputMask ] },
		@"inputs": @{ @"unit": NFKModelLiveUnitVideoFrame, @"inputs": @[], @"outputs": @[ NFKOutputMask ] },
		@"outputs": @{ @"unit": NFKModelLiveUnitVideoFrame, @"inputs": @[ NFKInputImage ] },
		@"sampleRate": @{ @"unit": NFKModelLiveUnitAudioChunk, @"inputs": @[ NFKInputAudio ], @"outputs": @[ NFKOutputAudio ],
						  @"chunkDuration": @0.02, @"channelCount": @1 },
		@"chunkDuration": @{ @"unit": NFKModelLiveUnitVideoFrame, @"inputs": @[ NFKInputImage ], @"outputs": @[ NFKOutputMask ],
							 @"chunkDuration": @0.02 },
		@"timings[0]": @{ @"unit": NFKModelLiveUnitVideoFrame, @"inputs": @[ NFKInputImage ], @"outputs": @[ NFKOutputMask ],
						  @"timings": @[ [self timingAtWidth:0 height:0 milliseconds:5] ] },
		@"lookahead": @{ @"unit": NFKModelLiveUnitVideoFrame, @"inputs": @[ NFKInputImage ], @"outputs": @[ NFKOutputMask ],
						 @"lookahead": @-1 },
	};
	for (NSString *field in cases) {
		NSError *error = nil;
		XCTAssertNil([NFKModelLiveProfile profileWithDictionary:cases[field] error:&error], @"%@", field);
		XCTAssertEqual(error.code, NFKModelCatalogErrorMalformedDescriptor, @"%@", field);
		XCTAssertTrue([error.localizedDescription containsString:field], @"%@: %@", field, error);
	}
}

#pragma mark Descriptors

- (void)testADescriptorCarriesItsProfileThroughItsDictionary
{
	NSDictionary *live = [self videoLiveWithTimings:@[ [self timingAtWidth:1280 height:720 milliseconds:8] ]];
	NSError *error = nil;
	NFKModelDescriptor *descriptor = [self descriptorNamed:@"matter" live:live error:&error];
	XCTAssertNotNil(descriptor, @"%@", error);
	XCTAssertEqualObjects(descriptor.liveProfile.unit, NFKModelLiveUnitVideoFrame);
	XCTAssertEqualObjects(descriptor.liveProfile.inputKeys, @[ NFKInputImage ]);
	XCTAssertEqualObjects(descriptor.dictionaryRepresentation[@"live"], descriptor.liveProfile.dictionaryRepresentation);
	XCTAssertEqualObjects([NFKModelDescriptor descriptorWithDictionary:descriptor.dictionaryRepresentation error:&error], descriptor,
						  @"%@", error);
	XCTAssertNil([self descriptorNamed:@"still" live:nil error:&error].liveProfile, @"%@", error);
	XCTAssertNil([self descriptorNamed:@"still" live:nil error:NULL].dictionaryRepresentation[@"live"]);
}

- (void)testAProfileMustNameTheDescriptorsPortsAndVariants
{
	NSDictionary *timing = [self timingAtWidth:1280 height:720 milliseconds:8];
	NSMutableDictionary *variantTiming = [timing mutableCopy];
	variantTiming[@"variant"] = @"q8";
	NSMutableDictionary *defaultTiming = [timing mutableCopy];
	defaultTiming[@"variant"] = @"fp16";
	NSDictionary<NSString *, NSDictionary *> *cases = @{
		@"live.inputs": @{ @"unit": NFKModelLiveUnitVideoFrame, @"inputs": @[ NFKInputAudio ], @"outputs": @[ NFKOutputMask ] },
		@"live.outputs": @{ @"unit": NFKModelLiveUnitVideoFrame, @"inputs": @[ NFKInputImage ], @"outputs": @[ NFKOutputDetections ] },
		@"names variant \"q8\"": [self videoLiveWithTimings:@[ variantTiming ]],
		@"repeats the Apple M3 Max timing": [self videoLiveWithTimings:@[ timing, defaultTiming ]],
	};
	for (NSString *reason in cases) {
		NSError *error = nil;
		XCTAssertNil([self descriptorNamed:@"matter" live:cases[reason] error:&error], @"%@", reason);
		XCTAssertEqual(error.code, NFKModelCatalogErrorMalformedDescriptor, @"%@", reason);
		XCTAssertTrue([error.localizedDescription containsString:reason], @"%@: %@", reason, error);
	}
}

#pragma mark Keeping up

- (void)testAVideoTimingAnswersForItsSizeOrTheNextLarger
{
	NSMutableDictionary *quantized = [[self timingAtWidth:1280 height:720 milliseconds:10] mutableCopy];
	quantized[@"variant"] = @"q4";
	NSDictionary *live = [self videoLiveWithTimings:@[ [self timingAtWidth:1280 height:720 milliseconds:15],
													   [self timingAtWidth:1920 height:1080 milliseconds:30], quantized ]];
	NFKModelLiveProfile *profile = [self descriptorNamed:@"matter" live:live error:NULL].liveProfile;
	XCTAssertEqual([profile timingsForVariant:nil].count, 2u);
	XCTAssertEqual([profile timingsForVariant:@"q4"].count, 1u);
	XCTAssertEqualObjects([profile keepsUpWithRate:60 variant:nil width:1280 height:720 chip:@"Apple M3 Max"], @YES, @"15 ms within 16.7");
	XCTAssertEqualObjects([profile keepsUpWithRate:60 variant:nil width:1920 height:1080 chip:@"Apple M3 Max"], @NO);
	XCTAssertEqualObjects([profile keepsUpWithRate:30 variant:nil width:1920 height:1080 chip:@"Apple M3 Max"], @YES, @"30 ms within 33.3");
	XCTAssertEqualObjects([profile keepsUpWithRate:60 variant:nil width:960 height:540 chip:@"Apple M3 Max"], @YES,
						  @"a smaller frame reads the smallest measured size above it");
	XCTAssertEqualObjects([profile keepsUpWithRate:100 variant:@"q4" width:1280 height:720 chip:@"Apple M3 Max"], @YES);
	XCTAssertNil([profile keepsUpWithRate:30 variant:nil width:3840 height:2160 chip:@"Apple M3 Max"], @"no timing covers 4K");
	XCTAssertNil([profile keepsUpWithRate:30 variant:nil width:1280 height:720 chip:@"Apple M1"], @"no timing on that chip");
	XCTAssertNil([profile keepsUpWithRate:0 variant:nil width:1280 height:720 chip:@"Apple M3 Max"]);
}

- (void)testAnAudioTimingKeepsUpBelowItsChunksDuration
{
	NSDictionary *timing = @{ @"chip": @"Apple M3 Max", @"milliseconds": @6, @"measured": @"2026-11-06" };
	NSDictionary *live = @{ @"unit": NFKModelLiveUnitAudioChunk, @"inputs": @[ NFKInputAudio ], @"outputs": @[ NFKOutputAudio ],
							@"chunkDuration": @0.01, @"sampleRate": @48000, @"channelCount": @1, @"timings": @[ timing ] };
	NFKModelLiveProfile *profile = [NFKModelLiveProfile profileWithDictionary:live error:NULL];
	XCTAssertEqualObjects([profile keepsUpWithRate:0 variant:nil width:0 height:0 chip:@"Apple M3 Max"], @YES, @"6 ms per 10 ms chunk");
	NSMutableDictionary *slow = [live mutableCopy];
	slow[@"timings"] = @[ @{ @"chip": @"Apple M3 Max", @"milliseconds": @10, @"measured": @"2026-11-06" } ];
	profile = [NFKModelLiveProfile profileWithDictionary:slow error:NULL];
	XCTAssertEqualObjects([profile keepsUpWithRate:0 variant:nil width:0 height:0 chip:@"Apple M3 Max"], @NO, @"real time leaves no room");
}

#pragma mark Catalog

- (void)testTheCatalogListsItsVisibleLiveDescriptorsByUnit
{
	NSDictionary *audio = @{ @"unit": NFKModelLiveUnitAudioChunk, @"inputs": @[ NFKInputImage ], @"outputs": @[ NFKOutputMask ],
							 @"chunkDuration": @0.02, @"sampleRate": @16000, @"channelCount": @1 };
	NSMutableDictionary *standIn = [[self modelNamed:@"a-stand-in" live:[self videoLiveWithTimings:@[]]] mutableCopy];
	standIn[@"standIn"] = @YES;
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	[catalog addDescriptors:@[ [self descriptorNamed:@"z-video" live:[self videoLiveWithTimings:@[]] error:NULL],
							   [self descriptorNamed:@"b-video" live:[self videoLiveWithTimings:@[]] error:NULL],
							   [self descriptorNamed:@"m-audio" live:audio error:NULL],
							   [self descriptorNamed:@"still" live:nil error:NULL],
							   [NFKModelDescriptor descriptorWithDictionary:standIn error:NULL] ]];
	XCTAssertEqualObjects([[catalog liveDescriptorsForUnit:NFKModelLiveUnitVideoFrame] valueForKey:@"identifier"],
						  (@[ @"b-video", @"z-video" ]));
	XCTAssertEqualObjects([[catalog liveDescriptorsForUnit:NFKModelLiveUnitAudioChunk] valueForKey:@"identifier"], @[ @"m-audio" ]);
	XCTAssertEqualObjects([[catalog liveDescriptorsForUnit:nil] valueForKey:@"identifier"], (@[ @"b-video", @"m-audio", @"z-video" ]));
}

- (void)testTheVisionEnginesThatAnswerAFrameAreLive
{
	NFKModelCatalog *catalog = NFKModelCatalog.sharedCatalog;
	NSDictionary<NSString *, NSArray<NSString *> *> *expected = @{ @"vision-segmentation": @[ NFKOutputMask ],
																	@"vision-pose": @[ NFKOutputPose ],
																	@"vision-face": @[ NFKOutputDetections, NFKOutputPose ] };
	for (NSString *identifier in expected) {
		NFKModelLiveProfile *profile = [catalog descriptorWithIdentifier:identifier].liveProfile;
		XCTAssertEqualObjects(profile.unit, NFKModelLiveUnitVideoFrame, @"%@", identifier);
		XCTAssertEqualObjects(profile.inputKeys, @[ NFKInputImage ], @"%@", identifier);
		XCTAssertEqualObjects(profile.outputKeys, expected[identifier], @"%@", identifier);
		XCTAssertFalse(profile.isStateful, @"%@", identifier);
	}
	XCTAssertNil([catalog descriptorWithIdentifier:@"vision-text"].liveProfile);
}

@end
