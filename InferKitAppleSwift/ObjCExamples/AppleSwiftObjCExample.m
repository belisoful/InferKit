//
//  AppleSwiftObjCExample.m
//  InferKitAppleSwiftObjCExamples
//
//  The Objective-C half. This package exists so these APIs are reachable from here at all: the
//  analyzer is a Swift actor and the two Vision requests ship with no VN* header, so an Objective-C
//  app cannot call any of them directly.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
@import InferKitAppleSwift;

@interface AppleSwiftObjCExample : XCTestCase
@end

@implementation AppleSwiftObjCExample

- (void)testObjectiveCReadsADocumentsStructure
{
	// Vision's document reader has no Objective-C header. Wrapped as a backend, it takes the same
	// request every other engine takes.
	NFKVisionDocumentBackend *backend = [[NFKVisionDocumentBackend alloc] init];
	XCTAssertEqualObjects(backend.backendIdentifier, @"vision-document");
	XCTAssertTrue([backend.supportedInputKeys containsObject:NFKInputImage]);

	// NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
	// result.text is the transcript; result.structured carries paragraphs, lists, and tables.
}

- (void)testObjectiveCJudgesTheLens
{
	NFKVisionSmudgeBackend *backend = [[NFKVisionSmudgeBackend alloc] init];
	XCTAssertEqualObjects(backend.backendIdentifier, @"vision-smudge");
	// The verdict is one NFKClassification labeled "smudge"; its confidence is how dirty the lens was.
}

- (void)testObjectiveCTranscribesWithTheAnalyzer
{
	NFKSpeechAnalyzerBackend *backend =
		[[NFKSpeechAnalyzerBackend alloc] initWithLocale:[NSLocale localeWithLocaleIdentifier:@"en-US"]];
	XCTAssertEqualObjects(backend.locale.localeIdentifier, @"en-US");
	XCTAssertFalse(backend.isReady, @"prepare installs the locale's assets and is what makes it ready");

	// The locale lists are asynchronous, because the system answers them that way.
	XCTestExpectation *listed = [self expectationWithDescription:@"installed locales"];
	[NFKSpeechAnalyzerBackend installedLocalesWithCompletionHandler:^(NSArray<NSLocale *> *locales) {
		(void)locales;
		[listed fulfill];
	}];
	[self waitForExpectations:@[ listed ] timeout:30.0];
}

- (void)testObjectiveCOpensLiveCaptions
{
	NFKSpeechAnalyzerBackend *backend =
		[[NFKSpeechAnalyzerBackend alloc] initWithLocale:[NSLocale localeWithLocaleIdentifier:@"en-US"]];
	// id<NFKLiveStream> captions = NFKInferenceOpenLiveStream(backend, nil, &error);
	// NFKInferenceResult *heard = [captions resultForUnit:unit error:&error];
	// heard.segments are final; [heard outputForKey:NFKOutputTentativeText] may still change.
	XCTAssertTrue([backend respondsToSelector:@selector(openLiveStreamWithParameters:error:)]);
	XCTAssertTrue([backend.supportedOutputKeys containsObject:NFKOutputTentativeText]);
}

- (void)testObjectiveCReachesTheAnalyzerThroughDiscovery
{
	NSError *error = nil;
	id<NFKInferenceBackend> backend = [NFKDynamicBackend backendForCapability:NFKCapabilityTranscription
																		error:&error];
	XCTAssertNotNil(backend, @"%@", error);
	XCTAssertEqualObjects(backend.backendIdentifier, @"apple-speech-analyzer");
}

@end
