//
//  NFKVisionClassificationBackend.h
//  InferKit
//

#ifndef NFKVisionClassificationBackend_h
#define NFKVisionClassificationBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKVisionClassificationBackend
	@abstract   Names what an image shows, through Apple's Vision framework.
	@discussion NFKInputImage in, NFKClassifications under NFKOutputClassifications, ordered by
				confidence. Vision's taxonomy is a fixed list of more than a thousand scene and
				object terms, so a caller reads labels rather than choosing a class set.

				Vision scores every term in the taxonomy, and most of them score low. minimumConfidence
				drops the rest; 0.1 by default, 0 to receive them all. Precision and recall are the
				more principled filter where a caller knows which it wants, and
				minimumRecallForPrecision applies Vision's own curve instead.

				A model with a chosen class set belongs on `NFKCoreMLBackend` or an MLX model.
				Introduced in InferKit 0.4.0.
*/
@interface NFKVisionClassificationBackend : NSObject <NFKInferenceBackend>

/*! Terms below this confidence are dropped. 0.1 by default. */
@property (nonatomic) double minimumConfidence;

/*! When above 0, keep the terms that reach this recall at the precision Vision measured, and ignore
	minimumConfidence. 0 by default. */
@property (nonatomic) double minimumRecallForPrecision;

+ (instancetype)backend;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKVisionClassificationBackend_h */
