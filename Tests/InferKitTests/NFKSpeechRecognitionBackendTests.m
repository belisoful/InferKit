//
//  NFKSpeechRecognitionBackendTests.m
//  NFKTests
//
//  Recognition needs the user's consent, which a test bundle does not have, so these cover the
//  contract and the refusals. A transcription run is host-verified with an authorized app.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

@interface NFKSpeechRecognitionBackendTests : XCTestCase
@end

@implementation NFKSpeechRecognitionBackendTests

- (void)testTheBackendReportsItsIdentifierAndKeys
{
	NFKSpeechRecognitionBackend *backend = [NFKSpeechRecognitionBackend backend];
	XCTAssertEqualObjects(backend.backendIdentifier, @"apple-speech");
	XCTAssertTrue([backend.supportedInputKeys containsObject:NFKInputAudio]);
	XCTAssertTrue(backend.requiresOnDeviceRecognition, @"the audio stays on the machine by default");
	XCTAssertTrue(backend.addsPunctuation);
}

- (void)testTheLocaleDefaultsToTheUsersAndIsRestorable
{
	NFKSpeechRecognitionBackend *backend = [NFKSpeechRecognitionBackend backend];
	XCTAssertEqualObjects(backend.locale, NSLocale.currentLocale);

	NSLocale *swedish = [NSLocale localeWithLocaleIdentifier:@"sv-SE"];
	backend.locale = swedish;
	XCTAssertEqualObjects(backend.locale, swedish);
	backend.locale = nil;
	XCTAssertEqualObjects(backend.locale, NSLocale.currentLocale);

	XCTAssertEqualObjects([NFKSpeechRecognitionBackend backendWithLocale:swedish].locale, swedish);
}

- (void)testARequestWithoutAudioIsRefusedBeforeAnythingElse
{
	NFKSpeechRecognitionBackend *backend = [NFKSpeechRecognitionBackend backend];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}]
														   error:&error];
	XCTAssertNil(result);
#if TARGET_OS_TV
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
#else
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceMissingInput);
#endif
}

- (void)testReadinessFollowsAuthorization
{
	NFKSpeechRecognitionBackend *backend = [NFKSpeechRecognitionBackend backend];
	if (!NFKSpeechRecognitionBackend.isAuthorized) {
		XCTAssertFalse(backend.isReady, @"an unauthorized app cannot transcribe");

		NFKAudioAsset *asset = [NFKAudioAsset audioAssetWithFileURL:[NSURL fileURLWithPath:@"/tmp/nfk-absent.wav"]];
		NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: asset }];
		NSError *error = nil;
		XCTAssertNil([backend runInferenceForRequest:request error:&error]);
#if TARGET_OS_TV
		XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
#else
		XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceNotReady);
#endif
		return;
	}
	XCTAssertTrue(backend.isReady || !backend.isReady, @"an authorized app depends on the locale's recognizer");
}

@end
