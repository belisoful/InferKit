//
//  NFKVisionFeaturePrintBackend.h
//  InferKit
//

#ifndef NFKVisionFeaturePrintBackend_h
#define NFKVisionFeaturePrintBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKVisionFeaturePrintBackend
	@abstract   Embeds an image through Apple's Vision framework, for image-to-image similarity.
	@discussion NFKInputImage in, the vector under NFKOutputEmbedding as an array of numbers, the
				shape every other embedding backend returns. Two prints compare by distance, so a
				caller ranks a library against a query with no download and no model to convert.

				Vision embeds images only. Text-to-image retrieval needs a model with a text tower,
				which is what the MLX CLIP, SigLIP 2, and MetaCLIP backends are for.
				Introduced in InferKit 0.4.0.
*/
@interface NFKVisionFeaturePrintBackend : NSObject <NFKInferenceBackend>

+ (instancetype)backend;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKVisionFeaturePrintBackend_h */
