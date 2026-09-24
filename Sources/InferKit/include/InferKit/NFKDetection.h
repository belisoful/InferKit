//
//  NFKDetection.h
//  InferKit
//

#ifndef NFKDetection_h
#define NFKDetection_h

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import "NFKQuadrilateral.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKDetection
	@abstract   One detected object: a class, a confidence, and a bounding box.
	@discussion An object-detection backend returns an
				NSArray<NFKDetection *> under NFKOutputDetections. The bounding box is normalized to
				0...1 in the input image's coordinate space, origin top-left, so a consumer scales it
				to any display size. classIndex is the model's raw class id; label is the human-readable
				name when the backend has a class list, nil when it returns indices only.

				The type archives: a consumer that records a result per frame writes an array of them
				through NSKeyedArchiver with secure coding on, and reads it back with
				unarchivedObjectOfClasses:. Conformance introduced in InferKit 0.4.0.
*/
@interface NFKDetection : NSObject <NSCopying, NSSecureCoding>

/*! The class name, or nil when the backend returns a class index only. */
@property (nonatomic, readonly, nullable, copy) NSString *label;

/*! The model's class index. */
@property (nonatomic, readonly) NSInteger classIndex;

/*! The detection confidence in 0...1. */
@property (nonatomic, readonly) double confidence;

/*! The bounding box, normalized to 0...1 in the input image, origin top-left. */
@property (nonatomic, readonly) CGRect boundingBox;

/*! The four corners, where the engine reports a shape a box cannot hold: a page at an angle, a
	barcode in perspective. nil where the engine reports a box only. The box is the corners'
	bounding box in that case, so a caller that wants either always has one. Introduced in
	InferKit 0.4.0. */
@property (nonatomic, readonly, nullable) NFKQuadrilateral *quadrilateral;

+ (instancetype)detectionWithLabel:(nullable NSString *)label
						classIndex:(NSInteger)classIndex
						confidence:(double)confidence
					   boundingBox:(CGRect)boundingBox;

- (instancetype)initWithLabel:(nullable NSString *)label
				   classIndex:(NSInteger)classIndex
				   confidence:(double)confidence
				  boundingBox:(CGRect)boundingBox;

/*! A detection whose engine reports four corners; the bounding box is taken from them.
	Introduced in InferKit 0.4.0. */
+ (instancetype)detectionWithLabel:(nullable NSString *)label
						classIndex:(NSInteger)classIndex
						confidence:(double)confidence
					 quadrilateral:(NFKQuadrilateral *)quadrilateral;

- (instancetype)initWithLabel:(nullable NSString *)label
				   classIndex:(NSInteger)classIndex
				   confidence:(double)confidence
				  boundingBox:(CGRect)boundingBox
				quadrilateral:(nullable NFKQuadrilateral *)quadrilateral NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKDetection_h */
