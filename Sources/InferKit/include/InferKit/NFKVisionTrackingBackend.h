//
//  NFKVisionTrackingBackend.h
//  InferKit
//

#ifndef NFKVisionTrackingBackend_h
#define NFKVisionTrackingBackend_h

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@enum       NFKVisionTrackingKind
	@abstract   What the backend follows across frames.
	@constant   NFKVisionTrackingKindObject  A region the caller named, followed frame by frame.
	@constant   NFKVisionTrackingKindTrajectory  Objects moving on a parabola, found without being
				named.
*/
typedef NS_ENUM(NSInteger, NFKVisionTrackingKind) {
	NFKVisionTrackingKindObject		= 0,
	NFKVisionTrackingKindTrajectory	= 1,
};

/*!
	@class      NFKVisionTrackingBackend
	@abstract   Follows something across frames through Apple's Vision framework.
	@discussion This backend holds state, which the others do not. Vision's tracking requests are
				sequence requests: each frame's answer depends on the frames before it, so the
				backend keeps the sequence between calls. One instance follows one sequence, and a
				caller wanting two at once makes two.

				Object tracking: name the region with `startTrackingBoundingBox:`, then submit each
				frame under NFKInputImage. Every run returns the region's new position as a single
				NFKDetection under NFKOutputDetections, labeled `tracked`, and the backend carries it
				forward as the input for the next frame. `isTracking` is NO before the first call and
				after the region is lost.

				Trajectory detection: submit frames under NFKInputImage and read NFKOutputStructured,
				which carries `trajectories`, each with `detectedPoints` and `projectedPoints` as
				flat x, y arrays in the contract's geometry and `equationCoefficients`, the three
				numbers of the parabola. Nothing is reported until `trajectoryLength` frames have
				been seen, so early frames return an empty reading rather than an error.

				`reset` ends the sequence and forgets the state, which is what a caller does when the
				clip changes. Introduced in InferKit 0.4.0.
*/
@interface NFKVisionTrackingBackend : NSObject <NFKInferenceBackend>

/*! What to follow. An object by default. */
@property (nonatomic) NFKVisionTrackingKind kind;

/*! YES once a region is being followed and not yet lost. Object tracking only. */
@property (nonatomic, readonly) BOOL isTracking;

/*! The frames a trajectory needs before it is reported. 5 by default, which is Vision's minimum. */
@property (nonatomic) NSInteger trajectoryLength;

/*! The rate the submitted frames are timed at. 30 by default. Vision times a trajectory and a still
	image carries no timestamp, so the backend stamps each frame in the order it arrives. A caller
	whose frames are not evenly spaced sets this to their real rate. */
@property (nonatomic) double framesPerSecond;

/*! Names the region to follow, normalized 0...1 with the origin at the top left, as every other
	engine reports one. Starts a new sequence. */
- (void)startTrackingBoundingBox:(CGRect)boundingBox;

/*! Ends the sequence and forgets the state. */
- (void)reset;

+ (instancetype)backend;

/*! A backend for one kind. */
+ (instancetype)backendWithKind:(NFKVisionTrackingKind)kind;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKVisionTrackingBackend_h */
