//
//  NFKDetection.h
//  InferKit
//

#ifndef NFKDetection_h
#define NFKDetection_h

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKDetection
	@abstract   One detected object: a class, a confidence, and a bounding box.
	@discussion An object-detection backend returns an
				NSArray<NFKDetection *> under NFKOutputDetections. The bounding box is normalized to
				0...1 in the input image's coordinate space, origin top-left, so a consumer scales it
				to any display size. classIndex is the model's raw class id; label is the human-readable
				name when the backend has a class list, nil when it returns indices only.
*/
@interface NFKDetection : NSObject <NSCopying>

/*! The class name, or nil when the backend returns a class index only. */
@property (nonatomic, readonly, nullable, copy) NSString *label;

/*! The model's class index. */
@property (nonatomic, readonly) NSInteger classIndex;

/*! The detection confidence in 0...1. */
@property (nonatomic, readonly) double confidence;

/*! The bounding box, normalized to 0...1 in the input image, origin top-left. */
@property (nonatomic, readonly) CGRect boundingBox;

+ (instancetype)detectionWithLabel:(nullable NSString *)label
						classIndex:(NSInteger)classIndex
						confidence:(double)confidence
					   boundingBox:(CGRect)boundingBox;

- (instancetype)initWithLabel:(nullable NSString *)label
				   classIndex:(NSInteger)classIndex
				   confidence:(double)confidence
				  boundingBox:(CGRect)boundingBox NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKDetection_h */
