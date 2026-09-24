//
//  NFKSpeechSynthesisBackend.h
//  InferKit
//

#ifndef NFKSpeechSynthesisBackend_h
#define NFKSpeechSynthesisBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKSpeechSynthesisBackend
	@abstract   Speaks text with the system's own voices, through AVFoundation.
	@discussion The same contract `NFKRemoteSpeechBackend` and the MLX voices answer, from voices the
				machine already has. There is nothing to download and no key to hold, which is what
				makes it the sensible default for an app that wants speech at all rather than a
				particular voice.

				NFKInputPrompt in. An NFKAudioAsset under NFKOutputAudio, a WAV file written to the
				temporary directory, which is the shape every other speech backend returns.

				`voiceIdentifier` names a voice; `availableVoices` lists what the machine has, each
				entry an identifier a caller can set. Without one, the voice follows `language`, and
				without that, the user's own setting. `rate`, `pitch`, and `volume` are AVFoundation's
				own scales.

				A chosen voice, a cloned voice, or a voice the machine does not have is Kokoro or
				Chatterbox in InferKitMLX. Personal Voice belongs to the person, so the system only
				offers it once they have allowed it; this backend reports it in availableVoices when
				they have. Introduced in InferKit 0.4.0.
*/
@interface NFKSpeechSynthesisBackend : NSObject <NFKInferenceBackend>

/*! The voice to speak with, by identifier. nil follows `language`. */
@property (nonatomic, copy, nullable) NSString *voiceIdentifier;

/*! The language to pick a voice for, BCP-47. nil follows the user's setting. */
@property (nonatomic, copy, nullable) NSString *language;

/*! Speaking rate, AVFoundation's scale. 0 uses the system default. */
@property (nonatomic) double rate;

/*! Pitch multiplier, 0.5 to 2. 1 by default. */
@property (nonatomic) double pitch;

/*! Volume, 0 to 1. 1 by default. */
@property (nonatomic) double volume;

/*! The voices this machine has, as identifiers. */
@property (class, nonatomic, readonly, copy) NSArray<NSString *> *availableVoices;

+ (instancetype)backend;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKSpeechSynthesisBackend_h */
