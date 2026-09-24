//
//  NFKVisionSegmentationBackend.h
//  InferKit
//

#ifndef NFKVisionSegmentationBackend_h
#define NFKVisionSegmentationBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@enum       NFKVisionSegmentationKind
	@abstract   Which mask the backend produces.
	@constant   NFKVisionSegmentationKindForegroundInstance  The photographic subject, whatever it is
				(macOS 14, iOS 17, tvOS 17).
	@constant   NFKVisionSegmentationKindPerson  People only, at a chosen quality (macOS 12, iOS 15,
				tvOS 15).
	@constant   NFKVisionSegmentationKindAttentionSaliency  Where a viewer looks first.
	@constant   NFKVisionSegmentationKindObjectnessSaliency  Where the objects are.
	@constant   NFKVisionSegmentationKindPersonInstances  Up to four people, masked together
				(macOS 14, iOS 17, tvOS 17). Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKVisionSegmentationKind) {
	NFKVisionSegmentationKindForegroundInstance	= 0,
	NFKVisionSegmentationKindPerson				= 1,
	NFKVisionSegmentationKindAttentionSaliency	= 2,
	NFKVisionSegmentationKindObjectnessSaliency	= 3,
	NFKVisionSegmentationKindPersonInstances	= 4,
};

/*!
	@enum       NFKVisionSegmentationQuality
	@abstract   The person mask's quality, traded against speed. Other kinds ignore it.
*/
typedef NS_ENUM(NSInteger, NFKVisionSegmentationQuality) {
	NFKVisionSegmentationQualityAccurate	= 0,
	NFKVisionSegmentationQualityBalanced	= 1,
	NFKVisionSegmentationQualityFast		= 2,
};

/*!
	@class      NFKVisionSegmentationBackend
	@abstract   Produces a mask for an image through Apple's Vision framework.
	@discussion The same output the MLX matting and segmentation models produce, from a system model
				that needs no weights: NFKInputImage in, a CVPixelBuffer mask under NFKOutputMask.
				kind chooses what the mask covers.

				The subject and person kinds postdate the core's floor, so a run below their OS fails
				with kNFKError_InferenceUnsupported and isReady is NO. The two saliency kinds run
				wherever the core runs. A mask from Vision is soft, single-channel, and sized by
				Vision rather than by the input, so a caller scales it to the plate.

				Where the model matters more than the download, the MLX matting models give a finer
				matte on hair and glass, and BiRefNet covers subjects Vision's single model does not.
				Introduced in InferKit 0.4.0.
*/
@interface NFKVisionSegmentationBackend : NSObject <NFKInferenceBackend>

/*! What the mask covers. The photographic subject by default. */
@property (nonatomic) NFKVisionSegmentationKind kind;

/*! The person mask's quality. Accurate by default. */
@property (nonatomic) NFKVisionSegmentationQuality quality;

+ (instancetype)backend;

/*! A backend for one kind. */
+ (instancetype)backendWithKind:(NFKVisionSegmentationKind)kind;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKVisionSegmentationBackend_h */
