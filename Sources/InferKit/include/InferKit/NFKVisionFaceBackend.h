//
//  NFKVisionFaceBackend.h
//  InferKit
//

#ifndef NFKVisionFaceBackend_h
#define NFKVisionFaceBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKVisionFaceBackend
	@abstract   Finds faces and their landmarks through Apple's Vision framework.
	@discussion NFKInputImage in; one NFKDetection per face under NFKOutputDetections, labeled
				"face", and, when detectsLandmarks is YES, every landmark point under NFKOutputPose
				as an NFKKeypoint. Both are in the contract's geometry: normalized 0...1 with the
				origin at the top left.

				Vision normalizes a landmark inside the face's own box; the backend maps each point
				back into the image, so a caller draws landmarks and boxes in one space. index
				counts points within one face and restarts at zero for the next, which is how a
				caller splits the array.

				RetinaFace is the alternative where a chosen detector matters, and CodeFormer's
				photograph backend pairs a detector with restoration. Introduced in InferKit 0.4.0.
*/
@interface NFKVisionFaceBackend : NSObject <NFKInferenceBackend>

/*! YES to read each face's landmarks as well as its box. YES by default. */
@property (nonatomic) BOOL detectsLandmarks;

+ (instancetype)backend;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKVisionFaceBackend_h */
