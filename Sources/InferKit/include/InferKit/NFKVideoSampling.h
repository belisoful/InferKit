//
//  NFKVideoSampling.h
//  InferKit
//

#ifndef NFKVideoSampling_h
#define NFKVideoSampling_h

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKVideoSampling
	@abstract   Frames sampled from a clip, for showing a video to a model that reads images.
	@discussion A vision model answers a question about a clip from a handful of its frames, which
				is how the chat backends carry NFKInputVideo: the frames become images beside the
				prompt. The frames are taken at evenly spaced times through the clip, the first at
				half a step in so the sample is centered rather than starting on the first frame.
				Introduced in InferKit 0.3.0.
*/
@interface NFKVideoSampling : NSObject

- (instancetype)init NS_UNAVAILABLE;

/*!
	@method     framesOfVideoAtURL:count:error:
	@abstract   count frames spaced evenly through the clip, as CGImages, or nil with an error.
	@discussion Blocks while decoding; run it off the render thread. A clip shorter than count frames
				yields the frames it has. The images honor the track's preferred transform, so a
				clip shot rotated comes back upright.
*/
+ (nullable NSArray *)framesOfVideoAtURL:(NSURL *)url
								   count:(NSUInteger)count
								   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKVideoSampling_h */
