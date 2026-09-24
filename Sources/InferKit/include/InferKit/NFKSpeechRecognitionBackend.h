//
//  NFKSpeechRecognitionBackend.h
//  InferKit
//

#ifndef NFKSpeechRecognitionBackend_h
#define NFKSpeechRecognitionBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKSpeechRecognitionBackend
	@abstract   Transcribes recorded audio through Apple's speech recognizer.
	@discussion The same contract Whisper and Parakeet answer, from a system recognizer with no
				weights to download: an NFKAudioAsset or NSData under NFKInputAudio in, the
				transcription under NFKOutputText, and one NFKAudioSegment per spoken word under
				NFKOutputSegments with its time range and confidence.

				Recognition needs the user's consent, which the app asks for once through
				requestAuthorizationWithCompletionHandler: and the system remembers. Until it is
				granted, isReady is NO and a run fails with kNFKError_InferenceNotReady. An app
				also declares NSSpeechRecognitionUsageDescription in its Info.plist; without it the
				request is denied.

				requiresOnDeviceRecognition keeps the audio on the machine, which is the default
				here. Turning it off lets the recognizer use Apple's servers, which reaches
				languages the device has no model for and sends the audio off the device.
				Apple's recognizer transcribes; it does not translate. Whisper keeps translation
				and word timestamps for any locale, and Parakeet keeps token-level timing.

				Unavailable on tvOS, where Apple ships no speech recognizer: there isReady is NO and
				a run reports kNFKError_InferenceUnsupported. Introduced in InferKit 0.4.0.
*/
@interface NFKSpeechRecognitionBackend : NSObject <NFKInferenceBackend>

/*! The locale to transcribe. The user's current locale by default. Setting nil restores it. */
@property (nonatomic, copy, null_resettable) NSLocale *locale;

/*! YES to refuse anything that would leave the machine. YES by default. */
@property (nonatomic) BOOL requiresOnDeviceRecognition;

/*! YES to punctuate the transcription (macOS 13, iOS 16). YES by default. */
@property (nonatomic) BOOL addsPunctuation;

/*! YES once the user has allowed speech recognition for this app. */
@property (class, nonatomic, readonly, getter=isAuthorized) BOOL authorized;

/*! Asks the user for speech recognition, once per app. The handler runs on an arbitrary thread. */
+ (void)requestAuthorizationWithCompletionHandler:(void (^)(BOOL authorized))handler;

+ (instancetype)backend;

/*! A backend for one locale. */
+ (instancetype)backendWithLocale:(NSLocale *)locale;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKSpeechRecognitionBackend_h */
