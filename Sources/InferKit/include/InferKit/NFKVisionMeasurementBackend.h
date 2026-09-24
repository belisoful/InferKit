//
//  NFKVisionMeasurementBackend.h
//  InferKit
//

#ifndef NFKVisionMeasurementBackend_h
#define NFKVisionMeasurementBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@enum       NFKVisionMeasurementKind
	@abstract   Which reading the backend takes.
	@constant   NFKVisionMeasurementKindAesthetics  How well composed the picture is, and whether it
				is a utility shot such as a screenshot or a receipt (macOS 15, iOS 18, tvOS 18).
	@constant   NFKVisionMeasurementKindHorizon  The angle the horizon sits at, for straightening.
*/
typedef NS_ENUM(NSInteger, NFKVisionMeasurementKind) {
	NFKVisionMeasurementKindAesthetics	= 0,
	NFKVisionMeasurementKindHorizon		= 1,
};

/*!
	@class      NFKVisionMeasurementBackend
	@abstract   Takes a numeric reading of an image through Apple's Vision framework.
	@discussion NFKInputImage in, the reading under NFKOutputStructured, which is where a named number
				with no key of its own belongs.

				Aesthetics reports `overallScore`, from -1 to 1, and `isUtility`, which is YES for a
				screenshot, a receipt, or a document rather than a photograph. A gallery uses the
				score to rank and the flag to exclude.

				Horizon reports `angleRadians` and `transform`, the six numbers of the affine
				transform that levels the picture, in the order a, b, c, d, tx, ty. Both stay in
				Vision's own image space, where the origin is the bottom left, because the transform
				is applied to pixels and a flipped angle would disagree with it. This is the one
				reading in the toolkit that is not in the contract's top-left geometry, and it is
				stated here because nothing in the type says so. A picture with no visible horizon
				produces an empty reading rather than an error.

				Introduced in InferKit 0.4.0.
*/
@interface NFKVisionMeasurementBackend : NSObject <NFKInferenceBackend>

/*! Which reading to take. Aesthetics by default. */
@property (nonatomic) NFKVisionMeasurementKind kind;

+ (instancetype)backend;

/*! A backend for one reading. */
+ (instancetype)backendWithKind:(NFKVisionMeasurementKind)kind;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKVisionMeasurementBackend_h */
