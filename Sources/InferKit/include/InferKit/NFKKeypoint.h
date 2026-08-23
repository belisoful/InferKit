//
//  NFKKeypoint.h
//  InferKit
//

#ifndef NFKKeypoint_h
#define NFKKeypoint_h

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKKeypoint
	@abstract   One located body or object landmark: a position and a confidence.
	@discussion A pose-estimation backend returns an
				NSArray<NFKKeypoint *> under NFKOutputPose, one entry per joint. The position is
				normalized to 0...1 in the input image's coordinate space, origin top-left, so a
				consumer scales it to any display size. index is the model's joint index (for a COCO
				pose, 0 is the nose); name is the human-readable joint name when the backend has a
				joint list, nil when it returns indices only. A low confidence marks an occluded or
				uncertain joint.
*/
@interface NFKKeypoint : NSObject <NSCopying>

/*! The joint name, or nil when the backend returns a joint index only. */
@property (nonatomic, readonly, nullable, copy) NSString *name;

/*! The model's joint index. */
@property (nonatomic, readonly) NSInteger index;

/*! The joint position, normalized to 0...1 in the input image, origin top-left. */
@property (nonatomic, readonly) CGPoint position;

/*! The detection confidence in 0...1. */
@property (nonatomic, readonly) double confidence;

+ (instancetype)keypointWithName:(nullable NSString *)name
						   index:(NSInteger)index
						position:(CGPoint)position
					  confidence:(double)confidence;

- (instancetype)initWithName:(nullable NSString *)name
					  index:(NSInteger)index
				   position:(CGPoint)position
				 confidence:(double)confidence NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKKeypoint_h */
