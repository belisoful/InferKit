//
//  NFKVideoToolboxBackend.h
//  InferKit
//

#ifndef NFKVideoToolboxBackend_h
#define NFKVideoToolboxBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@enum       NFKVideoToolboxTask
	@abstract   Which frame processor the backend runs.
	@constant   NFKVideoToolboxTaskSuperResolution  Upscales one frame by scaleFactor (macOS 26,
				iOS 26).
	@constant   NFKVideoToolboxTaskFrameInterpolation  Synthesizes the frame between two (macOS 26,
				iOS 26, tvOS 26).
	@constant   NFKVideoToolboxTaskOpticalFlow  Estimates motion between two frames (macOS 15.4,
				iOS 26).
*/
typedef NS_ENUM(NSInteger, NFKVideoToolboxTask) {
	NFKVideoToolboxTaskSuperResolution		= 0,
	NFKVideoToolboxTaskFrameInterpolation	= 1,
	NFKVideoToolboxTaskOpticalFlow			= 2,
};

/*!
	@class      NFKVideoToolboxBackend
	@abstract   Runs Apple's neural video processors: upscaling, frame interpolation, optical flow.
	@discussion VideoToolbox performs on Apple silicon what the MLX video models perform on chosen
				weights, with no download. The backend puts all three behind the contract.

				Upscaling reads NFKInputImage and returns the larger frame under NFKOutputImage.
				Interpolation and flow read two frames under NFKInputImages, previous first, and
				return one frame under NFKOutputImage: the synthesized middle frame, or the motion
				field. Every processor reads a pixel format of its own, so a frame is converted on
				the way in and the result is converted back to BGRA. A flow field comes back at the
				processor's own resolution, which is smaller than the frame, so a caller scales it.

				The flow field is packed the way NFKMLXRAFT packs it, so a caller reads either
				engine the same way: red and green carry the horizontal and vertical components as
				`0.5 + component / (2 * flowScale)`, mid-gray is no motion, and blue is zero. Decode
				a channel with `(value - 0.5) * 2 * flowScale`.

				Every processor needs Apple silicon and a recent OS, and upscaling needs a model the
				system downloads once. isReady reports whether the configured task can run here, and
				prepare() starts that download and reports what it is waiting for, so a pending model
				is a backend that is not ready rather than a run that fails. A frame must be a size
				the processor accepts, and a source in an unsupported pixel format is refused by
				name rather than silently converted.

				Real-ESRGAN, SwinIR, and HAT are the alternatives for upscaling a still on chosen
				weights, RIFE for interpolation, and RAFT for flow. Introduced in InferKit 0.4.0.
*/
@interface NFKVideoToolboxBackend : NSObject <NFKInferenceBackend>

/*! Which processor runs. Upscaling by default. */
@property (nonatomic) NFKVideoToolboxTask task;

/*! The upscaling factor. 0 by default, which takes the smallest factor the machine offers; a
	factor it does not offer is refused by name. Other tasks ignore it. */
@property (nonatomic) NSInteger scaleFactor;

/*! The pixel range a packed flow channel covers, in pixels. 32 by default, as NFKMLXRAFT uses. */
@property (nonatomic) double flowScale;

+ (instancetype)backend;

/*! A backend for one task. */
+ (instancetype)backendWithTask:(NFKVideoToolboxTask)task;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKVideoToolboxBackend_h */
