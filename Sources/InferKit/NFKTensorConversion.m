//
//  NFKTensorConversion.m
//  InferKit
//

#import "NFKTensorConversion.h"

/*! The RGBA source index feeding output channel c under the given order. */
static NSUInteger NFKTensorSourceIndex(NFKTensorChannelOrder order, NSUInteger channel)
{
	if (order == NFKTensorChannelOrderBGRA) {
		static const NSUInteger bgra[4] = { 2, 1, 0, 3 };
		return channel < 4 ? bgra[channel] : channel;
	}
	return channel;
}

/*! The tensor element index for channel c at (x, y) under the spec's layout. */
static NSUInteger NFKTensorIndex(NFKTensorSpec spec, NSUInteger channel, NSUInteger x, NSUInteger y)
{
	if (spec.layout == NFKTensorLayoutHWC) {
		return (y * spec.width + x) * spec.channelCount + channel;
	}
	return channel * spec.height * spec.width + y * spec.width + x;
}

NFKTensorSpec NFKTensorSpecMake(NSUInteger width, NSUInteger height, NSUInteger channelCount)
{
	NFKTensorSpec spec = {
		.width = width,
		.height = height,
		.channelCount = channelCount,
		.layout = NFKTensorLayoutCHW,
		.channelOrder = NFKTensorChannelOrderRGBA,
		.mean = { 0.0f, 0.0f, 0.0f, 0.0f },
		.scale = { 1.0f, 1.0f, 1.0f, 1.0f },
	};
	return spec;
}

NSUInteger NFKTensorElementCount(NFKTensorSpec spec)
{
	return spec.channelCount * spec.width * spec.height;
}

void NFKInterleavedToTensor(const float *interleaved, float *tensor, NFKTensorSpec spec)
{
	for (NSUInteger y = 0; y < spec.height; y++) {
		for (NSUInteger x = 0; x < spec.width; x++) {
			const float *pixel = interleaved + (y * spec.width + x) * 4;
			for (NSUInteger c = 0; c < spec.channelCount; c++) {
				float value = pixel[NFKTensorSourceIndex(spec.channelOrder, c)];
				tensor[NFKTensorIndex(spec, c, x, y)] = (value - spec.mean[c]) * spec.scale[c];
			}
		}
	}
}

void NFKTensorToInterleaved(const float *tensor, float *interleaved, NFKTensorSpec spec)
{
	for (NSUInteger y = 0; y < spec.height; y++) {
		for (NSUInteger x = 0; x < spec.width; x++) {
			float *pixel = interleaved + (y * spec.width + x) * 4;
			pixel[0] = 0.0f;
			pixel[1] = 0.0f;
			pixel[2] = 0.0f;
			pixel[3] = 1.0f;
			for (NSUInteger c = 0; c < spec.channelCount; c++) {
				float value = tensor[NFKTensorIndex(spec, c, x, y)];
				float denormalized = spec.scale[c] != 0.0f ? value / spec.scale[c] + spec.mean[c] : spec.mean[c];
				pixel[NFKTensorSourceIndex(spec.channelOrder, c)] = denormalized;
			}
		}
	}
}
