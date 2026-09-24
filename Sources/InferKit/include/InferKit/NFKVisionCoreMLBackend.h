//
//  NFKVisionCoreMLBackend.h
//  InferKit
//

#ifndef NFKVisionCoreMLBackend_h
#define NFKVisionCoreMLBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

@class MLModel;

NS_ASSUME_NONNULL_BEGIN

/*!
	@enum       NFKVisionCropAndScale
	@abstract   How an image is fitted to the model's input size.
	@constant   NFKVisionCropAndScaleCenterCrop  Scale the short side and crop the centre.
	@constant   NFKVisionCropAndScaleScaleToFit  Squash the whole image into the input, changing its
				aspect ratio.
	@constant   NFKVisionCropAndScaleScaleToFill  Scale the long side and crop nothing, letting the
				image extend past the input.
*/
typedef NS_ENUM(NSInteger, NFKVisionCropAndScale) {
	NFKVisionCropAndScaleCenterCrop		= 0,
	NFKVisionCropAndScaleScaleToFit		= 1,
	NFKVisionCropAndScaleScaleToFill	= 2,
};

/*!
	@class      NFKVisionCoreMLBackend
	@abstract   Runs a Core ML vision model through Vision, which prepares the image for it.
	@discussion `NFKCoreMLBackend` hands a model the tensors a caller built. This one hands Vision
				the model and lets Vision do the fitting: scaling, cropping, colour conversion, and
				the orientation the image carries. A model converted with an image input is easier to
				drive this way, and the crop rule is stated rather than implied.

				NFKInputImage in. What comes out depends on what the model emits, which Vision reports
				as typed observations:

				classifications → NFKOutputClassifications;
				objects with boxes → NFKOutputDetections, labeled and in the contract's geometry;
				a pixel buffer → NFKOutputMask;
				anything else → the feature under its own name, as the model named it.

				A model that emits several of those fills several keys. Introduced in InferKit 0.4.0.
*/
@interface NFKVisionCoreMLBackend : NSObject <NFKInferenceBackend>

/*! How the image is fitted to the model's input. Centre crop by default, as Vision defaults. */
@property (nonatomic) NFKVisionCropAndScale cropAndScale;

/*! Classifications below this confidence are dropped. 0 keeps them all, which is the default. */
@property (nonatomic) double minimumConfidence;

/*! A backend over a loaded Core ML model. Returns nil when Vision refuses the model, which it does
	for a model whose input is not an image. */
+ (nullable instancetype)backendWithModel:(MLModel *)model error:(NSError **)error;

/*! A backend over a compiled model on disk (an .mlmodelc directory). */
+ (nullable instancetype)backendWithCompiledModelURL:(NSURL *)url error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKVisionCoreMLBackend_h */
