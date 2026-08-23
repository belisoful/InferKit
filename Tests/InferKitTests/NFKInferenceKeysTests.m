//
//  NFKInferenceKeysTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKRemoteBackend.h>

@interface NFKInferenceKeysTests : XCTestCase
@end

@implementation NFKInferenceKeysTests

- (void)testTheStandardKeyValues
{
	XCTAssertEqualObjects(NFKInputPrompt, @"prompt");
	XCTAssertEqualObjects(NFKInputNegativePrompt, @"negativePrompt");
	XCTAssertEqualObjects(NFKParameterSeed, @"seed");
	XCTAssertEqualObjects(NFKParameterGuidanceScale, @"guidanceScale");
	XCTAssertEqualObjects(NFKParameterStrength, @"strength");
	XCTAssertEqualObjects(NFKOutputImage, @"image");
	XCTAssertEqualObjects(NFKOutputVideo, @"video");
	XCTAssertEqualObjects(NFKInputAudio, @"audio");
	XCTAssertEqualObjects(NFKOutputAudio, @"audio");
	XCTAssertEqualObjects(NFKParameterSampleRate, @"sampleRate");
	XCTAssertEqualObjects(NFKParameterChannelCount, @"channelCount");
}

- (void)testARequestBuiltFromTheVocabularyReadsBack
{
	NSDictionary *inputs = @{
		NFKInputPrompt: @"a red car",
		NFKInputNegativePrompt: @"blurry",
	};
	NSDictionary *parameters = @{
		NFKParameterSeed: @42,
		NFKParameterSteps: @30,
		NFKParameterGuidanceScale: @7.5,
	};
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:inputs parameters:parameters];

	XCTAssertEqualObjects([request inputForKey:NFKInputPrompt], @"a red car");
	XCTAssertEqualObjects([request inputForKey:NFKInputNegativePrompt], @"blurry");
	XCTAssertEqualObjects([request parameterForKey:NFKParameterSeed], @42);
	XCTAssertEqualObjects([request parameterForKey:NFKParameterGuidanceScale], @7.5);
}

- (void)testThePromptAndTextKeysAlignWithTheRemoteBackend
{
	// A request built with the shared vocabulary interoperates with the remote backend, whose
	// prompt/text keys carry the same values.
	XCTAssertEqualObjects(NFKInputPrompt, NFKRemoteBackendPromptKey);
	XCTAssertEqualObjects(NFKOutputText, NFKRemoteBackendTextKey);
}

@end
