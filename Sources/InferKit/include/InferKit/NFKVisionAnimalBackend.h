//
//  NFKVisionAnimalBackend.h
//  InferKit
//

#ifndef NFKVisionAnimalBackend_h
#define NFKVisionAnimalBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKVisionAnimalBackend
	@abstract   Finds cats and dogs in an image, through Apple's Vision framework.
	@discussion NFKInputImage in, one NFKDetection per animal under NFKOutputDetections, labeled with
				Vision's identifier ("Cat", "Dog") and boxed in the contract's geometry: normalized
				0...1 with the origin at the top left.

				Vision recognizes the animals its own model was trained for, which is two species.
				supportedIdentifiers reports what the installed revision actually finds, since the
				list is the model's rather than the API's. A detector over a chosen class set is
				`NFKCoreMLBackend` or one of the MLX detectors. Introduced in InferKit 0.4.0.
*/
@interface NFKVisionAnimalBackend : NSObject <NFKInferenceBackend>

/*! The animals the installed Vision revision recognizes, or an empty array when it cannot say. */
@property (class, nonatomic, readonly, copy) NSArray<NSString *> *supportedIdentifiers;

+ (instancetype)backend;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKVisionAnimalBackend_h */
