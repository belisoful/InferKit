//
//  NFKVideoAsset.h
//  InferKit
//

#ifndef NFKVideoAsset_h
#define NFKVideoAsset_h

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKVideoAsset
	@abstract   A clip that flows through inference as a video input or output.
	@discussion A video is a media file: a generated clip a remote
				model returns as an MP4 URL, or a source clip a video-to-video model reads. The
				file is the identity; duration, frame rate, and dimensions are optional metadata a
				producer fills when known (0 or CGSizeZero when not). A consumer opens fileURL with
				AVFoundation to decode or play it.
*/
@interface NFKVideoAsset : NSObject <NSCopying>

/*! The clip's media file. */
@property (nonatomic, readonly, nullable) NSURL *fileURL;

/*! The clip's duration in seconds, or 0 when unknown. */
@property (nonatomic, readonly) double durationSeconds;

/*! The clip's frame rate, or 0 when unknown. */
@property (nonatomic, readonly) double framesPerSecond;

/*! The clip's pixel dimensions, or CGSizeZero when unknown. */
@property (nonatomic, readonly) CGSize dimensions;

+ (instancetype)videoAssetWithFileURL:(NSURL *)fileURL;

+ (instancetype)videoAssetWithFileURL:(nullable NSURL *)fileURL
					  durationSeconds:(double)durationSeconds
					   framesPerSecond:(double)framesPerSecond
							dimensions:(CGSize)dimensions;

- (instancetype)initWithFileURL:(nullable NSURL *)fileURL
				durationSeconds:(double)durationSeconds
				 framesPerSecond:(double)framesPerSecond
					  dimensions:(CGSize)dimensions NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKVideoAsset_h */
