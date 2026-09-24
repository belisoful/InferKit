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
