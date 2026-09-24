//
//  NFKSoundClassificationBackend.h
//  InferKit
//

#ifndef NFKSoundClassificationBackend_h
#define NFKSoundClassificationBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKSoundClassificationBackend
	@abstract   Names the sounds in a recording, through Apple's Sound Analysis framework.
	@discussion Apple ships a classifier over several hundred everyday sounds: speech, laughter,
				applause, a dog, a door, rain. It needs no weights and no download, which is what
				makes it worth having beside the MLX audio tagger.

				NFKInputAudio in (an NFKAudioAsset or a file URL). Out: one NFKAudioSegment per
				window under NFKOutputSegments, labeled with the sound and carrying its confidence
				and time range, and the whole clip's best guesses under NFKOutputClassifications,
				ordered by the confidence they reached at any point.

				The classifier reads a window at a time, so a long recording yields many segments.
				`windowSeconds` widens or narrows them; 0 keeps the classifier's own default. Only
				sounds above `minimumConfidence` are reported, 0.3 by default, because the classifier
				scores every class in its taxonomy on every window.

				A chosen sound model, rather than Apple's list, is `NFKCoreMLBackend` with a Create ML
				sound classifier. Introduced in InferKit 0.4.0.
*/
@interface NFKSoundClassificationBackend : NSObject <NFKInferenceBackend>

/*! Sounds below this confidence are dropped. 0.3 by default. */
@property (nonatomic) double minimumConfidence;

/*! The window the classifier reads, in seconds. 0 uses the classifier's own default. */
@property (nonatomic) double windowSeconds;

/*! The sounds the installed classifier knows. */
@property (class, nonatomic, readonly, copy) NSArray<NSString *> *knownSounds;

+ (instancetype)backend;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKSoundClassificationBackend_h */
