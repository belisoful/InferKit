//
//  NFKBackendToolTests.m
//  InferKitTests
//
//  A backend as a tool: the default description and schema each modality gets, the request the
//  model's arguments become, and what the model reads of each kind of result.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKBackendTool.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKAudioAsset.h>
#import <InferKit/NFKErrors.h>
#import <CoreVideo/CoreVideo.h>

@interface NFKToolTestBackend : NSObject <NFKInferenceBackend>
@property (nonatomic, assign) BOOL ready;
@property (nonatomic, assign) NSInteger prepareCount;
@property (nonatomic, strong, nullable) NFKInferenceRequest *lastRequest;
@property (nonatomic, copy, nullable) NSDictionary<NSString *, id> *replyOutputs;
@property (nonatomic, copy, nullable) NSString *modelName;
@end

@implementation NFKToolTestBackend

- (BOOL)isReady
{
	return self.ready;
}

- (NSString *)backendIdentifier
{
	return @"tool-test";
}

- (BOOL)prepareWithError:(NSError **)error
{
	self.prepareCount += 1;
	self.ready = YES;
	return YES;
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	self.lastRequest = request;
	if (self.replyOutputs == nil) {
		*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceBackendFailure
								 userInfo:@{ NSLocalizedDescriptionKey: @"the model is out of memory" }];
		return nil;
	}
	return [NFKInferenceResult resultWithOutputs:self.replyOutputs];
}

@end

@interface NFKBackendToolTests : XCTestCase
@property (nonatomic, strong) NFKToolTestBackend *backend;
@end

@implementation NFKBackendToolTests

- (void)setUp
{
	[super setUp];
	self.backend = [[NFKToolTestBackend alloc] init];
	self.backend.ready = YES;
	self.backend.replyOutputs = @{ NFKOutputText: @"ok" };
}

#pragma mark Defaults

- (void)testTheDefaultDescriptionSaysWhatWhenAndWhatComesBack
{
	self.backend.modelName = @"qwen3.5-4b";
	NFKBackendTool *tool = [NFKBackendTool toolWithBackend:self.backend name:@"ask_qwen" outputModality:NFKModalityText
												   purpose:@"the user's text must stay on this machine"];
	XCTAssertEqualObjects(tool.name, @"ask_qwen");
	XCTAssertTrue([tool.toolDescription containsString:@"\"qwen3.5-4b\""], @"%@", tool.toolDescription);
	XCTAssertTrue([tool.toolDescription containsString:@"Use it when the user's text must stay on this machine."]);
	XCTAssertTrue([tool.toolDescription containsString:@"no access to this conversation"]);
	XCTAssertTrue([tool.toolDescription containsString:@"Returns the model's reply"]);
	XCTAssertEqualObjects(tool.parameters[@"required"], @[ NFKInputPrompt ]);
	XCTAssertNotNil(tool.parameters[@"properties"][@"system"]);
}

- (void)testEachModalityHasADefaultForWhenAndWhatComesBack
{
	for (NSNumber *modality in @[ @(NFKModalityText), @(NFKModalityAudio), @(NFKModalityImage), @(NFKModalityVideo) ]) {
		NSString *general = [NFKBackendTool defaultDescriptionForOutputModality:modality.integerValue modelName:@"m" purpose:nil];
		XCTAssertTrue([general containsString:@"Use it when "], @"%@", general);
		XCTAssertTrue([general containsString:@"Returns "], @"%@", general);
		NSDictionary *schema = [NFKBackendTool defaultParametersForOutputModality:modality.integerValue];
		XCTAssertEqualObjects(schema[@"type"], @"object");
		XCTAssertEqualObjects(schema[@"required"], @[ NFKInputPrompt ]);
		for (NSDictionary *property in [schema[@"properties"] allValues]) {
			XCTAssertGreaterThan([property[@"description"] length], 0u, @"every argument tells the model what it means");
		}
	}
	NSString *audio = [NFKBackendTool defaultDescriptionForOutputModality:NFKModalityAudio modelName:@"kokoro" purpose:nil];
	XCTAssertTrue([audio containsString:@"goes to the application"], @"a model told the audio reaches the user does not try to repeat it");
}

- (void)testTheModelIsNamedByTheBackendWhenTheCallerDoesNot
{
	NFKBackendTool *byIdentifier = [NFKBackendTool toolWithBackend:self.backend name:@"t" outputModality:NFKModalityText purpose:nil];
	XCTAssertTrue([byIdentifier.toolDescription containsString:@"\"tool-test\""]);
	NFKBackendTool *byName = [NFKBackendTool toolWithBackend:self.backend name:@"t" outputModality:NFKModalityText modelName:@"Qwen 3.5" purpose:nil];
	XCTAssertTrue([byName.toolDescription containsString:@"\"Qwen 3.5\""]);
}

#pragma mark Requests

- (void)testArgumentsBecomeInputsParametersAndASystemTurn
{
	NFKBackendTool *tool = [NFKBackendTool toolWithBackend:self.backend name:@"t" outputModality:NFKModalityText purpose:nil];
	tool.fixedParameters = @{ NFKParameterMaxTokens: @256, NFKParameterTemperature: @0.2 };
	NSDictionary *arguments = @{ NFKInputPrompt: @"Summarize this.", @"system": @"Be brief.", NFKParameterTemperature: @0.7 };
	NFKToolResult *result = [tool invokeWithArguments:arguments];
	XCTAssertFalse(result.isError, @"%@", result.text);

	NFKInferenceRequest *request = self.backend.lastRequest;
	XCTAssertEqual(request.outputModality, NFKModalityText);
	XCTAssertEqualObjects(request.prompt, @"Summarize this.");
	NSArray *messages = @[ @{ @"role": @"system", @"content": @"Be brief." }, @{ @"role": @"user", @"content": @"Summarize this." } ];
	XCTAssertEqualObjects(request.messages, messages);
	XCTAssertEqualObjects(request.parameters[NFKParameterMaxTokens], @256, @"the caller's fixed settings ride along");
	XCTAssertEqualObjects(request.parameters[NFKParameterTemperature], @0.7, @"the model's own argument wins over a fixed one");
	XCTAssertNil(request.parameters[@"system"]);
}

- (void)testAnAudioToolPassesItsLengthAsTheParameter
{
	NFKBackendTool *tool = [NFKBackendTool toolWithBackend:self.backend name:@"speak" outputModality:NFKModalityAudio purpose:nil];
	[tool invokeWithArguments:@{ NFKInputPrompt: @"Hello there.", NFKParameterDurationSeconds: @4 }];
	XCTAssertEqual(self.backend.lastRequest.outputModality, NFKModalityAudio);
	XCTAssertEqualObjects(self.backend.lastRequest.prompt, @"Hello there.");
	XCTAssertEqualObjects(self.backend.lastRequest.parameters[NFKParameterDurationSeconds], @4);
	XCTAssertNil(self.backend.lastRequest.messages, @"without a system instruction the prompt goes alone");
}

- (void)testABackendNotYetLoadedIsPreparedFirst
{
	self.backend.ready = NO;
	NFKBackendTool *tool = [NFKBackendTool toolWithBackend:self.backend name:@"t" outputModality:NFKModalityText purpose:nil];
	XCTAssertFalse([tool invokeWithArguments:@{ NFKInputPrompt: @"x" }].isError);
	XCTAssertEqual(self.backend.prepareCount, 1);
}

- (void)testTheCallersBuilderAndDescriberReplaceTheDefaults
{
	NFKBackendTool *tool = [[NFKBackendTool alloc] initWithBackend:self.backend name:@"lookup" description:@"Looks a word up."
														 parameters:@{ @"type": @"object", @"properties": @{ @"word": @{ @"type": @"string" } } }
													 outputModality:NFKModalityText];
	tool.requestBuilder = ^NFKInferenceRequest *(NSDictionary *arguments, NSError **error) {
		return [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: [@"Define: " stringByAppendingString:arguments[@"word"]] }
										   parameters:nil outputModality:NFKModalityText];
	};
	tool.resultDescriber = ^NFKToolResult *(NFKInferenceResult *result) {
		return [NFKToolResult resultWithText:[@"Definition: " stringByAppendingString:result.text]];
	};
	NFKToolResult *result = [tool invokeWithArguments:@{ @"word": @"tensor" }];
	XCTAssertEqualObjects(self.backend.lastRequest.prompt, @"Define: tensor");
	XCTAssertEqualObjects(result.text, @"Definition: ok");
}

#pragma mark Results

- (void)testAFailedRunReachesTheModelAsAnErrorResult
{
	self.backend.replyOutputs = nil;
	NFKBackendTool *tool = [NFKBackendTool toolWithBackend:self.backend name:@"t" outputModality:NFKModalityText purpose:nil];
	NFKToolResult *result = [tool invokeWithArguments:@{ NFKInputPrompt: @"x" }];
	XCTAssertTrue(result.isError);
	XCTAssertEqualObjects(result.text, @"t failed: the model is out of memory");
}

- (void)testGeneratedAudioIsDescribedToTheModelAndKeptForTheApplication
{
	NFKAudioAsset *clip = [NFKAudioAsset audioAssetWithFileURL:nil durationSeconds:3.5 sampleRate:24000 channelCount:1];
	self.backend.replyOutputs = @{ NFKOutputAudio: clip };
	NFKBackendTool *tool = [NFKBackendTool toolWithBackend:self.backend name:@"speak" outputModality:NFKModalityAudio purpose:nil];
	NFKToolResult *result = [tool invokeWithArguments:@{ NFKInputPrompt: @"Hello." }];
	XCTAssertEqualObjects(result.text, @"Generated 3.5 seconds of audio at 24000 Hz, 1 channel.");
	XCTAssertEqual(result.outputs[NFKOutputAudio], clip);
	XCTAssertEqualObjects(result.imagePNGs, @[]);
}

- (void)testAGeneratedImageIsShownToTheModel
{
	CVPixelBufferRef pixelBuffer = NULL;
	CVPixelBufferCreate(kCFAllocatorDefault, 4, 3, kCVPixelFormatType_32BGRA,
						(__bridge CFDictionaryRef)@{ (id)kCVPixelBufferIOSurfacePropertiesKey: @{} }, &pixelBuffer);
	self.backend.replyOutputs = @{ NFKOutputImage: (__bridge id)pixelBuffer };
	CVPixelBufferRelease(pixelBuffer);
	NFKBackendTool *tool = [NFKBackendTool toolWithBackend:self.backend name:@"draw" outputModality:NFKModalityImage purpose:nil];
	NFKToolResult *result = [tool invokeWithArguments:@{ NFKInputPrompt: @"a red square" }];
	XCTAssertEqualObjects(result.text, @"Generated a 4 by 3 image.");
	XCTAssertEqual(result.imagePNGs.count, 1u);
	const uint8_t *bytes = result.imagePNGs.firstObject.bytes;
	XCTAssertEqual(bytes[1], 'P', @"the image is shown as a PNG");
}

- (void)testStructuredRepliesAreReadAsSortedJSON
{
	self.backend.replyOutputs = @{ NFKOutputText: @"{…}", NFKOutputStructured: @{ @"b": @2, @"a": @1 } };
	NFKBackendTool *tool = [NFKBackendTool toolWithBackend:self.backend name:@"t" outputModality:NFKModalityText purpose:nil];
	XCTAssertEqualObjects([tool invokeWithArguments:@{ NFKInputPrompt: @"x" }].text, @"{\"a\":1,\"b\":2}");
}

@end
