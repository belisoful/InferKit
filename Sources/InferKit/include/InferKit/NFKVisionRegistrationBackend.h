//
//  NFKVisionRegistrationBackend.h
//  InferKit
//

#ifndef NFKVisionRegistrationBackend_h
#define NFKVisionRegistrationBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@enum       NFKVisionRegistrationKind
	@abstract   The kind of alignment to solve for.
	@constant   NFKVisionRegistrationKindTranslational  A shift, as an affine transform.
	@constant   NFKVisionRegistrationKindHomographic  A perspective warp, as a 3 by 3 matrix.
*/
typedef NS_ENUM(NSInteger, NFKVisionRegistrationKind) {
	NFKVisionRegistrationKindTranslational	= 0,
	NFKVisionRegistrationKindHomographic	= 1,
};

/*!
	@class      NFKVisionRegistrationBackend
	@abstract   Works out how one frame moved relative to another, through Apple's Vision framework.
	@discussion Two frames under NFKInputImages, the reference first and the moving one second, and
				the alignment under NFKOutputStructured. The translational kind reports `transform`
				as the six numbers of an affine transform, in the order a, b, c, d, tx, ty; the
				homographic kind reports `warpTransform` as nine numbers in row-major order.

				The transform maps the second frame onto the first, so applying it to the moving
				frame lines it up with the reference. Vision reports translation in pixels rather
				than normalized units, which is what a caller applies to the pixels.

				Stabilization, burst alignment, and panorama stitching are what this is for. Motion
				per pixel rather than per frame is `NFKVideoToolboxBackend`'s optical flow.
				Introduced in InferKit 0.4.0.
*/
@interface NFKVisionRegistrationBackend : NSObject <NFKInferenceBackend>

/*! Which alignment to solve for. Translational by default. */
@property (nonatomic) NFKVisionRegistrationKind kind;

+ (instancetype)backend;

/*! A backend for one kind. */
+ (instancetype)backendWithKind:(NFKVisionRegistrationKind)kind;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKVisionRegistrationBackend_h */
