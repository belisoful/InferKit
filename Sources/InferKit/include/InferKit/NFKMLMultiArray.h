//
//  NFKMLMultiArray.h
//  InferKit
//

#ifndef NFKMLMultiArray_h
#define NFKMLMultiArray_h

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import "NFKTensorConversion.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@header     NFKMLMultiArray
	@abstract   Bridges an RGBA image to a Core ML MLMultiArray tensor and back.
	@discussion Core ML models whose input is a normalized planar
				tensor (most diffusion and vision models) take an MLMultiArray, not an image. These
				functions complete the image-to-tensor path: NFKTensorConversion lays out and
				normalizes an RGBA float image into a planar buffer, and these wrap that buffer in a
				float32 MLMultiArray of the shape the spec describes, and read one back.

				The array is shaped [1, channels, height, width] for a CHW spec and
				[1, height, width, channels] for HWC, matching the planar buffer's memory order.
				Reading back expects a contiguous float32 array, which a model output normally is.
*/

/*! An MLMultiArray for the RGBA float image (0...1, tightly packed), or nil with an error. */
MLMultiArray * _Nullable NFKMultiArrayFromInterleaved(const float *interleaved,
													  NFKTensorSpec spec,
													  NSError * _Nullable * _Nullable error);

/*! Writes a contiguous float32 MLMultiArray back to an RGBA float image; NO for another data type. */
BOOL NFKInterleavedFromMultiArray(MLMultiArray *array, float *interleaved, NFKTensorSpec spec);

NS_ASSUME_NONNULL_END

#endif /* NFKMLMultiArray_h */
