//
//  NFKAudioAsset.h
//  InferKit
//

#ifndef NFKAudioAsset_h
#define NFKAudioAsset_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKAudioAsset
	@abstract   A clip that flows through inference as an audio input or output.
	@discussion Audio is a media file: a generated clip a
				text-to-speech or music model returns as a file URL, or a source clip a
				transcription or audio-to-audio model reads. The file is the identity; duration,
				sample rate, and channel count are optional metadata a producer fills when known (0
				when not). A consumer opens fileURL with AVFoundation to decode or play it.

				Audio mirrors NFKVideoAsset: a file-based value type. A backend that works with
				in-memory samples uses NSData (PCM) or an AVAudioPCMBuffer under the audio key
				instead, the same way an image input is a CVPixelBuffer or a texture.
*/
@interface NFKAudioAsset : NSObject <NSCopying>

/*! The clip's media file. */
@property (nonatomic, readonly, nullable) NSURL *fileURL;

/*! The clip's duration in seconds, or 0 when unknown. */
@property (nonatomic, readonly) double durationSeconds;

/*! The clip's sample rate in hertz, or 0 when unknown. */
@property (nonatomic, readonly) double sampleRate;

/*! The clip's channel count, or 0 when unknown. */
@property (nonatomic, readonly) NSInteger channelCount;

+ (instancetype)audioAssetWithFileURL:(NSURL *)fileURL;

+ (instancetype)audioAssetWithFileURL:(nullable NSURL *)fileURL
					  durationSeconds:(double)durationSeconds
						   sampleRate:(double)sampleRate
						  channelCount:(NSInteger)channelCount;

- (instancetype)initWithFileURL:(nullable NSURL *)fileURL
				durationSeconds:(double)durationSeconds
					 sampleRate:(double)sampleRate
					channelCount:(NSInteger)channelCount NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKAudioAsset_h */
