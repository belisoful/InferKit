//
//  NFKModelPresetTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

@interface NFKModelPresetTests : XCTestCase
@end

@implementation NFKModelPresetTests

- (void)testTheSeparatorsAreAtAndTwoUnderscores
{
	XCTAssertEqualObjects(NFKModelPresetSeparator, @"@");
	XCTAssertEqualObjects(NFKModelToolNameSeparator, @"__");
	XCTAssertEqualObjects(NFKModelPresetModelName(@"whisper", @"fast"), @"whisper@fast");
}

- (void)testANameSplitsAtItsLastSeparatorWhereThePresetIsDefined
{
	BOOL (^defined)(NSString *, NSString *) = ^BOOL(NSString *model, NSString *preset) {
		return [preset isEqualToString:@"fast"];
	};
	NSString *model = nil, *preset = nil;
	XCTAssertTrue(NFKModelNameSplitAtPreset(@"whisper@fast", defined, &model, &preset));
	XCTAssertEqualObjects(model, @"whisper");
	XCTAssertEqualObjects(preset, @"fast");

	XCTAssertTrue(NFKModelNameSplitAtPreset(@"a@b@fast", defined, &model, &preset));
	XCTAssertEqualObjects(model, @"a@b");
	XCTAssertEqualObjects(preset, @"fast");

	XCTAssertFalse(NFKModelNameSplitAtPreset(@"claude-3@20240620", defined, &model, &preset));
	XCTAssertEqualObjects(model, @"claude-3@20240620", @"a name with no defined preset passes whole");
	XCTAssertNil(preset);
}

- (void)testWithoutAPredicateEveryNonEmptySplitHappens
{
	NSString *model = nil, *preset = nil;
	XCTAssertTrue(NFKModelNameSplitAtPreset(@"claude-3@20240620", nil, &model, &preset));
	XCTAssertEqualObjects(model, @"claude-3");
	XCTAssertEqualObjects(preset, @"20240620");
	XCTAssertFalse(NFKModelNameSplitAtPreset(@"whisper", nil, &model, &preset));
	XCTAssertEqualObjects(model, @"whisper");
	XCTAssertFalse(NFKModelNameSplitAtPreset(@"@fast", nil, &model, &preset));
	XCTAssertFalse(NFKModelNameSplitAtPreset(@"whisper@", nil, &model, &preset));
	XCTAssertTrue(NFKModelNameSplitAtPreset(@"whisper@fast", nil, NULL, NULL));
}

- (void)testAToolNameKeepsLettersDigitsUnderscoresAndHyphens
{
	XCTAssertEqualObjects(NFKModelToolName(@"whisper@fast"), @"whisper__fast");
	XCTAssertEqualObjects(NFKModelToolName(@"mlx-community/Qwen2.5"), @"mlx-community_Qwen2_5");
	XCTAssertEqualObjects(NFKModelToolName(@"a@b@fast"), @"a__b__fast");
	NSString *longName = [@"" stringByPaddingToLength:80 withString:@"m" startingAtIndex:0];
	XCTAssertEqual(NFKModelToolName(longName).length, 64u);
}

- (void)testAPresetRoundTripsThroughItsDictionary
{
	NSDictionary *dictionary = @{ @"id": @"fast", @"title": @"Fast", @"help": @"Fewer steps.", @"base": @"whisper",
								  @"loadOptions": @{ NFKModelLoadOptionQuantizationBits: @4 },
								  @"requestDefaults": @{ NFKParameterTemperature: @0 }, @"suggested": @YES,
								  @"x-badge": @"new" };
	NSError *error = nil;
	NFKModelPreset *preset = [NFKModelPreset presetWithDictionary:dictionary error:&error];
	XCTAssertNotNil(preset, @"%@", error);
	XCTAssertEqualObjects(preset.dictionaryRepresentation, dictionary);
	XCTAssertTrue(preset.changesLoadOptions);
	XCTAssertTrue(preset.isSuggested);
	XCTAssertEqualObjects(preset.extensions, (@{ @"x-badge": @"new" }));
	XCTAssertEqualObjects([preset modelNameForBaseModelIdentifier:nil], @"whisper@fast");

	XCTAssertNil([NFKModelPreset presetWithDictionary:@{ @"title": @"No id" } error:&error]);
	XCTAssertEqualObjects(error.localizedDescription, @"preset.id is missing");
}

- (void)testAPresetUnderADescriptorTakesTheDescriptorAsItsBase
{
	NSDictionary *dictionary = @{ @"id": @"whisper", @"family": @"whisper", @"engine": NFKModelEngineMLX,
								  @"presets": @[ @{ @"id": @"fast", @"requestDefaults": @{ @"beams": @1 } } ] };
	NFKModelDescriptor *descriptor = [NFKModelDescriptor descriptorWithDictionary:dictionary error:NULL];
	NFKModelPreset *fast = [descriptor presetWithIdentifier:@"fast"];
	XCTAssertNil(fast.baseModelIdentifier);
	XCTAssertFalse(fast.changesLoadOptions);
	XCTAssertEqualObjects(fast.title, @"fast");
	XCTAssertEqualObjects([fast modelNameForBaseModelIdentifier:descriptor.identifier], @"whisper@fast");
	XCTAssertEqualObjects(descriptor.dictionaryRepresentation[@"presets"],
						  (@[ @{ @"id": @"fast", @"title": @"fast", @"requestDefaults": @{ @"beams": @1 },
								 @"suggested": @NO } ]));
	XCTAssertNil([descriptor presetWithIdentifier:@"slow"]);
}

@end
