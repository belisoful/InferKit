//
//  NFKVisionContourBackend.h
//  InferKit
//

#ifndef NFKVisionContourBackend_h
#define NFKVisionContourBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKVisionContourBackend
	@abstract   Traces the edges in an image as paths, through Apple's Vision framework.
	@discussion NFKInputImage in, the outlines under NFKOutputStructured: `contours`, an array with
				one entry per traced outline, each a flat array of numbers x, y, x, y in the
				contract's geometry, normalized 0...1 with the origin at the top left. `topLevelCount`
				says how many of them are outermost, since Vision nests a hole inside the shape that
				contains it and this shape reports them flattened.

				Points rather than a CGPath, because a path is a Core Graphics object and a result
				travels between processes, into a plist, and over a wire. A caller builds a path from
				the numbers in two lines.

				`detail` trades points for fidelity, and `contrastAdjustment` pulls faint edges out of
				a flat image. A mask of the subject is `NFKVisionSegmentationBackend`; this traces
				what is already visible. Introduced in InferKit 0.4.0.
*/
@interface NFKVisionContourBackend : NSObject <NFKInferenceBackend>

/*! How finely to trace, 0 to 1. 0.5 by default; higher keeps more points. */
@property (nonatomic) double detail;

/*! Contrast applied before tracing. 2 by default. */
@property (nonatomic) double contrastAdjustment;

/*! YES to trace dark shapes on a light field rather than the reverse. NO by default. */
@property (nonatomic) BOOL detectsDarkOnLight;

/*! The largest number of points to report per contour. 0 reports them all, which is the default. */
@property (nonatomic) NSInteger maximumPointsPerContour;

+ (instancetype)backend;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKVisionContourBackend_h */
