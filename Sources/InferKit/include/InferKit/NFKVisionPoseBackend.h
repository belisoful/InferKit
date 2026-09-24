//
//  NFKVisionPoseBackend.h
//  InferKit
//

#ifndef NFKVisionPoseBackend_h
#define NFKVisionPoseBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@enum       NFKVisionPoseKind
	@abstract   Which pose the backend estimates.
	@constant   NFKVisionPoseKindBody  Nineteen body joints per person.
	@constant   NFKVisionPoseKindHand  Twenty-one joints per hand.
	@constant   NFKVisionPoseKindAnimal  Twenty-five joints per cat or dog (macOS 14, iOS 17,
				tvOS 17). Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKVisionPoseKind) {
	NFKVisionPoseKindBody	= 0,
	NFKVisionPoseKindHand	= 1,
	NFKVisionPoseKindAnimal	= 2,
};

/*!
	@class      NFKVisionPoseBackend
	@abstract   Estimates body or hand pose through Apple's Vision framework.
	@discussion NFKInputImage in, NFKKeypoints under NFKOutputPose, in the contract's geometry:
				normalized 0...1 with the origin at the top left. Each keypoint carries Vision's
				joint name.

				Vision finds every subject in the frame, and the keypoints of each follow one
				another in the array. index counts joints within one subject and restarts at zero
				for the next, so a caller splits the array on that. Vision reports a joint it cannot
				see with a low confidence rather than omitting it; a caller filters on confidence.

				ViTPose is the alternative where accuracy on hard poses matters more than the
				absence of a download. Introduced in InferKit 0.4.0.
*/
@interface NFKVisionPoseBackend : NSObject <NFKInferenceBackend>

/*! Body or hand. Body by default. */
@property (nonatomic) NFKVisionPoseKind kind;

/*! The most hands to find. 2 by default. The body kind ignores it. */
@property (nonatomic) NSInteger maximumHandCount;

+ (instancetype)backend;

/*! A backend for one kind. */
+ (instancetype)backendWithKind:(NFKVisionPoseKind)kind;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKVisionPoseBackend_h */
