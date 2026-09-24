//
//  NFKSoundClassificationBackend.m
//  InferKit
//

#import "NFKSoundClassificationBackend.h"
#import "NFKInferenceKeys.h"
#import "NFKInferenceRequest.h"
#import "NFKInferenceResult.h"
#import "NFKAudioAsset.h"
#import "NFKAudioSegment.h"
#import "NFKClassification.h"
#import "NFKErrors.h"
#import <SoundAnalysis/SoundAnalysis.h>

/// Collects the windows the analyzer reports. The analyzer calls back on its own queue and the
/// synchronous analyze returns once it is done, so the array needs no lock beyond that ordering.
@interface NFKSoundObserver : NSObject <SNResultsObserving>
@property (nonatomic, strong) NSMutableArray<SNClassificationResult *> *results;
@property (nonatomic, strong, nullable) NSError *failure;
@end

@implementation NFKSoundObserver

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_results = [NSMutableArray array];
	}
	return self;
}

- (void)request:(id<SNRequest>)request didProduceResult:(id<SNResult>)result
{
	if ([result isKindOfClass:SNClassificationResult.class]) {
		[self.results addObject:(SNClassificationResult *)result];
	}
}

- (void)request:(id<SNRequest>)request didFailWithError:(NSError *)error
{
	self.failure = error;
}

- (void)requestDidComplete:(id<SNRequest>)request
{
}

@end

@implementation NFKSoundClassificationBackend

+ (instancetype)backend
{
	return [[self alloc] init];
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_minimumConfidence = 0.3;
	}
	return self;
}

+ (NSArray<NSString *> *)knownSounds
{
	NSError *error = nil;
	SNClassifySoundRequest *request =
		[[SNClassifySoundRequest alloc] initWithClassifierIdentifier:SNClassifierIdentifierVersion1 error:&error];
	return request.knownClassifications ?: @[];
}

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"sound-analysis";
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithObject:NFKInputAudio];
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError **)outError
{
	NSURL *url = [self audioURLForRequest:request];
	if (url == nil) {
		[self failWithError:outError
					   code:kNFKError_InferenceMissingInput
					 reason:@"no audio file is set under NFKInputAudio"];
		return nil;
	}

	NSError *analyzerError = nil;
	SNAudioFileAnalyzer *analyzer = [[SNAudioFileAnalyzer alloc] initWithURL:url error:&analyzerError];
	if (analyzer == nil) {
		NSString *reason = analyzerError.localizedDescription ?: @"the audio file could not be read";
		[self failWithError:outError code:kNFKError_InferenceMissingInput reason:reason];
		return nil;
	}

	SNClassifySoundRequest *classify =
		[[SNClassifySoundRequest alloc] initWithClassifierIdentifier:SNClassifierIdentifierVersion1
															   error:&analyzerError];
	if (classify == nil) {
		NSString *reason = analyzerError.localizedDescription ?: @"the sound classifier is unavailable";
		[self failWithError:outError code:kNFKError_InferenceUnsupported reason:reason];
		return nil;
	}
	if (self.windowSeconds > 0.0) {
		classify.windowDuration = CMTimeMakeWithSeconds(self.windowSeconds, 600);
	}

	NFKSoundObserver *observer = [[NFKSoundObserver alloc] init];
	if (![analyzer addRequest:classify withObserver:observer error:&analyzerError]) {
		NSString *reason = analyzerError.localizedDescription ?: @"the classifier refused the file";
		[self failWithError:outError code:kNFKError_InferenceBackendFailure reason:reason];
		return nil;
	}
	[analyzer analyze];

	if (observer.failure != nil) {
		[self failWithError:outError
					   code:kNFKError_InferenceBackendFailure
					 reason:observer.failure.localizedDescription ?: @"the analysis failed"];
		return nil;
	}
	return [self resultFromResults:observer.results];
}

// Every window keeps its own label; the clip's summary is each sound at the highest confidence it
// reached anywhere, which is what a caller asking "what is in this recording" means.
- (NFKInferenceResult *)resultFromResults:(NSArray<SNClassificationResult *> *)results
{
	NSMutableArray<NFKAudioSegment *> *segments = [NSMutableArray array];
	NSMutableDictionary<NSString *, NSNumber *> *best = [NSMutableDictionary dictionary];

	for (SNClassificationResult *result in results) {
		SNClassification *top = result.classifications.firstObject;
		if (top == nil || top.confidence < self.minimumConfidence) {
			continue;
		}
		[segments addObject:[NFKAudioSegment segmentWithStartSeconds:CMTimeGetSeconds(result.timeRange.start)
														 endSeconds:CMTimeGetSeconds(CMTimeRangeGetEnd(result.timeRange))
															  label:top.identifier
														 confidence:top.confidence]];
		for (SNClassification *classification in result.classifications) {
			if (classification.confidence < self.minimumConfidence) {
				continue;
			}
			NSNumber *previous = best[classification.identifier];
			if (previous == nil || classification.confidence > previous.doubleValue) {
				best[classification.identifier] = @(classification.confidence);
			}
		}
	}

	NSArray<NSString *> *ranked = [best keysSortedByValueUsingComparator:^NSComparisonResult(NSNumber *a, NSNumber *b) {
		return [b compare:a];
	}];
	NSMutableArray<NFKClassification *> *classifications = [NSMutableArray arrayWithCapacity:ranked.count];
	for (NSString *identifier in ranked) {
		[classifications addObject:[NFKClassification classificationWithLabel:identifier
																   classIndex:classifications.count
																   confidence:best[identifier].doubleValue]];
	}

	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputSegments: segments,
													NFKOutputClassifications: classifications }];
}

- (nullable NSURL *)audioURLForRequest:(NFKInferenceRequest *)request
{
	id audio = [request inputForKey:NFKInputAudio];
	if ([audio isKindOfClass:NFKAudioAsset.class]) {
		return ((NFKAudioAsset *)audio).fileURL;
	}
	if ([audio isKindOfClass:NSURL.class]) {
		return audio;
	}
	return nil;
}

- (BOOL)failWithError:(NSError **)error code:(NSInteger)code reason:(NSString *)reason
{
	if (error != NULL) {
		*error = [NSError errorWithDomain:NFKInferenceErrorDomain
									 code:code
								 userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return NO;
}

@end
