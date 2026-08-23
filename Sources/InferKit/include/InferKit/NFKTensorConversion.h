//
//  NFKTensorConversion.h
//  InferKit
//

#ifndef NFKTensorConversion_h
#define NFKTensorConversion_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@enum       NFKTensorLayout
	@abstract   The memory order of a tensor's channel and spatial axes.
	@discussion CHW is planar: all of channel 0, then all of channel 1, and so on — the layout
				most Core ML and PyTorch image models expect. HWC is interleaved by pixel.
*/
typedef NS_ENUM(NSInteger, NFKTensorLayout) {
	NFKTensorLayoutCHW	= 0,
	NFKTensorLayoutHWC	= 1,
};

/*!
	@enum       NFKTensorChannelOrder
	@abstract   The channel order a tensor's channels are drawn from the RGBA source.
	@discussion RGBA keeps source order; BGRA swaps red and blue, which some models expect.
*/
typedef NS_ENUM(NSInteger, NFKTensorChannelOrder) {
	NFKTensorChannelOrderRGBA	= 0,
	NFKTensorChannelOrderBGRA	= 1,
};

/*!
	@struct     NFKTensorSpec
	@abstract   Describes how an RGBA image maps to a model's input tensor.
	@discussion The bridge reads a source pixel channel selected by channelOrder for each output
				channel c (0 based), applies out = (in - mean[c]) * scale[c] with in in 0...1, and
				writes it at the position layout dictates. channelCount is 1, 3, or 4. mean and
				scale are indexed by output channel. The inverse denormalizes and restores RGBA,
				filling absent channels with 0 and alpha with 1.
*/
typedef struct NFKTensorSpec {
	NSUInteger width;
	NSUInteger height;
	NSUInteger channelCount;
	NFKTensorLayout layout;
	NFKTensorChannelOrder channelOrder;
	float mean[4];
	float scale[4];
} NFKTensorSpec;

/*! A spec with identity normalization (mean 0, scale 1) for the given size, RGB CHW. */
NFKTensorSpec NFKTensorSpecMake(NSUInteger width, NSUInteger height, NSUInteger channelCount);

/*! The element count of a tensor for a spec: channelCount * width * height. */
NSUInteger NFKTensorElementCount(NFKTensorSpec spec);

/*!
	@function   NFKInterleavedToTensor
	@abstract   Converts a width*height RGBA float image (0...1, tightly packed) into a tensor.
	@discussion interleaved holds width*height*4 floats. tensor receives NFKTensorElementCount
				floats in the spec's layout, normalized per channel. No bounds are checked; the
				buffers must be sized for the spec.
*/
void NFKInterleavedToTensor(const float *interleaved, float *tensor, NFKTensorSpec spec);

/*!
	@function   NFKTensorToInterleaved
	@abstract   Converts a tensor back to a width*height RGBA float image (0...1, tightly packed).
	@discussion Denormalizes per channel and writes RGBA, filling channels the tensor does not
				carry with 0 and alpha with 1 when channelCount is under 4.
*/
void NFKTensorToInterleaved(const float *tensor, float *interleaved, NFKTensorSpec spec);

NS_ASSUME_NONNULL_END

#endif /* NFKTensorConversion_h */
