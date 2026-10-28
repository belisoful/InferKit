//
//  NFKAppleEngineTests.m
//  NFKTests
//
//  The three Apple frameworks the core gained beside Vision and VideoToolbox: sound classification,
//  speech synthesis, and text embedding. Each runs a system model, so these need no weights.
//

#import <XCTest/XCTest.h>
#import <AVFoundation/AVFoundation.h>
#import <InferKit/InferKit.h>

@interface NFKAppleEngineTests : XCTestCase
@end

@implementation NFKAppleEngineTests

#pragma mark Sound classification

- (void)testTheSoundBackendReportsItsContract
{
	NFKSoundClassificationBackend *backend = [NFKSoundClassificationBackend backend];
	XCTAssertTrue(backend.isReady);
	XCTAssertEqualObjects(backend.backendIdentifier, @"sound-analysis");
	XCTAssertTrue([backend.supportedInputKeys containsObject:NFKInputAudio]);
	XCTAssertEqualWithAccuracy(backend.minimumConfidence, 0.3, 1e-9);
}

- (void)testTheClassifierNamesTheSoundsItKnows
{
	NSArray<NSString *> *sounds = NFKSoundClassificationBackend.knownSounds;
	XCTAssertGreaterThan(sounds.count, (NSUInteger)100, @"Apple's classifier covers several hundred");
	XCTAssertTrue([sounds containsObject:@"speech"], @"speech is in every revision of the taxonomy");
}

- (void)testASpokenClipIsClassified
{
	// The synthesizer writes a file, which the classifier then reads: two of the new engines in one
	// pass, and no fixture to check in.
	NFKSpeechSynthesisBackend *voice = [NFKSpeechSynthesisBackend backend];
	if (!voice.isReady) {
		return;
	}
	NFKInferenceRequest *spoken =
		[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"The quick brown fox jumps over the lazy dog." }];
	NSError *error = nil;
	NFKInferenceResult *audio = [voice runInferenceForRequest:spoken error:&error];
	XCTAssertNotNil(audio, @"%@", error);

	NFKAudioAsset *asset = [audio outputForKey:NFKOutputAudio];
	XCTAssertNotNil(asset.fileURL);

	NFKSoundClassificationBackend *classifier = [NFKSoundClassificationBackend backend];
	classifier.minimumConfidence = 0.05;
	NFKInferenceResult *heard =
		[classifier runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: asset }]
									 error:&error];
	XCTAssertNotNil(heard, @"%@", error);
	if (heard != nil && heard.segments.count == 0 && NFKHardwareProfile.currentProfile.isVirtualMachine) {
		// A hosted runner is a virtual machine, where the classifier reports no window for a clip it
		// hears on real hardware. The clip's length is in the message, so the log says which of the
		// two engines went quiet.
		AVAudioFile *clip = [[AVAudioFile alloc] initForReading:asset.fileURL error:NULL];
		XCTSkip("the classifier heard nothing in this virtual machine; the synthesizer wrote %lld frames at %.0f Hz",
				(long long)clip.length, clip.fileFormat.sampleRate);
	}
	XCTAssertGreaterThan(heard.segments.count, (NSUInteger)0, @"a spoken clip fills at least one window");
	XCTAssertGreaterThan(heard.classifications.count, (NSUInteger)0);
}

- (void)testTheSoundBackendNeedsAFile
{
	NSError *error = nil;
	XCTAssertNil([[NFKSoundClassificationBackend backend]
				  runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error]);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceMissingInput);
}

#pragma mark Speech synthesis

- (void)testTheVoiceBackendReportsItsContract
{
	NFKSpeechSynthesisBackend *backend = [NFKSpeechSynthesisBackend backend];
	XCTAssertEqualObjects(backend.backendIdentifier, @"apple-speech-synthesis");
	XCTAssertTrue([backend.supportedInputKeys containsObject:NFKInputPrompt]);
	XCTAssertGreaterThan(NFKSpeechSynthesisBackend.availableVoices.count, (NSUInteger)0);
	XCTAssertTrue(backend.isReady);
}

- (void)testSpeakingWritesAnAudioFile
{
	NFKSpeechSynthesisBackend *backend = [NFKSpeechSynthesisBackend backend];
	if (!backend.isReady) {
		return;
	}
	NSError *error = nil;
	NFKInferenceResult *result =
		[backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hello." }]
								  error:&error];
	XCTAssertNotNil(result, @"%@", error);

	NFKAudioAsset *asset = [result outputForKey:NFKOutputAudio];
	XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:asset.fileURL.path]);
	AVAudioFile *file = [[AVAudioFile alloc] initForReading:asset.fileURL error:&error];
	XCTAssertNotNil(file, @"%@", error);
	XCTAssertGreaterThan(file.length, 0, @"the file holds samples");
}

- (void)testAChosenVoiceIsUsed
{
	NSString *identifier = NFKSpeechSynthesisBackend.availableVoices.firstObject;
	if (identifier == nil) {
		return;
	}
	NFKSpeechSynthesisBackend *backend = [NFKSpeechSynthesisBackend backend];
	backend.voiceIdentifier = identifier;
	XCTAssertEqualObjects(backend.voiceIdentifier, identifier);
	NSError *error = nil;
	XCTAssertNotNil([backend runInferenceForRequest:
					 [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"One." }] error:&error], @"%@", error);
}

/*! The loudest sample of a spoken file, or a negative number when it cannot be read. */
- (float)peakOfAudioAt:(NSURL *)fileURL
{
	AVAudioFile *file = [[AVAudioFile alloc] initForReading:fileURL commonFormat:AVAudioPCMFormatFloat32 interleaved:NO
													  error:NULL];
	AVAudioPCMBuffer *buffer = [[AVAudioPCMBuffer alloc] initWithPCMFormat:file.processingFormat
															  frameCapacity:(AVAudioFrameCount)file.length];
	if (file == nil || buffer == nil || ![file readIntoBuffer:buffer error:NULL]) {
		return -1;
	}
	float peak = 0;
	for (AVAudioFrameCount frame = 0; frame < buffer.frameLength; frame++) {
		peak = MAX(peak, fabsf(buffer.floatChannelData[0][frame]));
	}
	return peak;
}

// A hosted backend is reached through its requests alone, so a request's voice and volume override the
// backend's properties for that request.
- (void)testARequestSetsTheVoiceAndTheVolume
{
	NSString *identifier = NFKSpeechSynthesisBackend.availableVoices.firstObject;
	if (identifier == nil) {
		return;
	}
	NFKSpeechSynthesisBackend *backend = [NFKSpeechSynthesisBackend backend];
	backend.voiceIdentifier = @"com.example.no-such-voice";
	NSError *error = nil;
	NFKInferenceResult *spoken = [backend runInferenceForRequest:
		[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hello there." }
									parameters:@{ NFKParameterVoice: identifier }] error:&error];
	XCTAssertNotNil(spoken, @"%@", error);
	XCTAssertGreaterThan([self peakOfAudioAt:[[spoken outputForKey:NFKOutputAudio] fileURL]], 0.01f);

	NFKInferenceResult *silent = [backend runInferenceForRequest:
		[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hello there." }
									parameters:@{ NFKParameterVoice: identifier, NFKSpeechSynthesisParameterVolume: @0 }]
														   error:&error];
	XCTAssertNotNil(silent, @"%@", error);
	XCTAssertEqual([self peakOfAudioAt:[[silent outputForKey:NFKOutputAudio] fileURL]], 0.0f);
	XCTAssertEqualObjects(backend.voiceIdentifier, @"com.example.no-such-voice", @"the request leaves the property as it was");
}

- (void)testARequestsSpeechSettingsAreChecked
{
	NFKSpeechSynthesisBackend *backend = [NFKSpeechSynthesisBackend backend];
	NSArray<NSDictionary<NSString *, id> *> *refused = @[ @{ NFKParameterVoice: @3 }, @{ NFKParameterSourceLanguage: @YES },
														  @{ NFKSpeechSynthesisParameterRate: @1.5 },
														  @{ NFKSpeechSynthesisParameterPitch: @0.1 },
														  @{ NFKSpeechSynthesisParameterVolume: @"loud" },
														  @{ NFKParameterVoice: @"com.example.no-such-voice" } ];
	for (NSDictionary<NSString *, id> *parameters in refused) {
		NSError *error = nil;
		XCTAssertNil([backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"One." }
																				  parameters:parameters] error:&error],
					 @"%@", parameters);
		XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported, @"%@", parameters);
	}
	NSSet *keys = [NSSet setWithObjects:NFKParameterVoice, NFKParameterSourceLanguage, NFKSpeechSynthesisParameterRate,
				   NFKSpeechSynthesisParameterPitch, NFKSpeechSynthesisParameterVolume, nil];
	XCTAssertEqualObjects(backend.supportedParameterKeys, keys);
}

- (void)testSpeakingNeedsText
{
	NSError *error = nil;
	XCTAssertNil([[NFKSpeechSynthesisBackend backend]
				  runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error]);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceMissingInput);
}

#pragma mark Text embedding

- (void)testTheEmbeddingBackendReportsItsContract
{
	NFKTextEmbeddingBackend *backend = [NFKTextEmbeddingBackend backend];
	XCTAssertEqualObjects(backend.backendIdentifier, @"natural-language-embedding");
	XCTAssertTrue([backend.supportedInputKeys containsObject:NFKInputPrompt]);
}

- (void)testASentenceBecomesAVector
{
	NFKTextEmbeddingBackend *backend = [NFKTextEmbeddingBackend backendWithLanguage:@"en"];
	if (!backend.isReady) {
		return;   // Apple ships no sentence model for English on this machine
	}
	XCTAssertGreaterThan(backend.dimension, 0);

	NSError *error = nil;
	NFKInferenceResult *result =
		[backend runInferenceForRequest:
		 [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"A gannet dives for fish." }] error:&error];
	XCTAssertNotNil(result, @"%@", error);
	XCTAssertEqual(result.embedding.count, (NSUInteger)backend.dimension);
}

- (void)testCloserSentencesEmbedCloser
{
	NFKTextEmbeddingBackend *backend = [NFKTextEmbeddingBackend backendWithLanguage:@"en"];
	if (!backend.isReady) {
		return;
	}
	NSArray<NSNumber *> *bird = [self embedding:@"A gannet dives for fish." with:backend];
	NSArray<NSNumber *> *seabird = [self embedding:@"A seabird plunges into the sea." with:backend];
	NSArray<NSNumber *> *tax = [self embedding:@"Quarterly tax returns are due on Friday." with:backend];
	if (bird == nil || seabird == nil || tax == nil) {
		return;
	}
	XCTAssertGreaterThan([self cosineBetween:bird and:seabird], [self cosineBetween:bird and:tax],
						 @"two sentences about seabirds sit closer than one about tax");
}

- (void)testAnUnsupportedLanguageIsNotReady
{
	NFKTextEmbeddingBackend *backend = [NFKTextEmbeddingBackend backendWithLanguage:@"xx-nonexistent"];
	XCTAssertFalse(backend.isReady);
	NSError *error = nil;
	XCTAssertNil([backend runInferenceForRequest:
				  [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"text" }] error:&error]);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceNotReady);
}

- (nullable NSArray<NSNumber *> *)embedding:(NSString *)text with:(NFKTextEmbeddingBackend *)backend
{
	return [backend runInferenceForRequest:
			[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: text }] error:NULL].embedding;
}

- (double)cosineBetween:(NSArray<NSNumber *> *)a and:(NSArray<NSNumber *> *)b
{
	double dot = 0, na = 0, nb = 0;
	for (NSUInteger i = 0; i < MIN(a.count, b.count); i++) {
		double x = a[i].doubleValue, y = b[i].doubleValue;
		dot += x * y; na += x * x; nb += y * y;
	}
	return (na > 0 && nb > 0) ? dot / (sqrt(na) * sqrt(nb)) : 0;
}

@end
