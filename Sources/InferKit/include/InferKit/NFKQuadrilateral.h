//
//  NFKQuadrilateral.h
//  InferKit
//

#ifndef NFKQuadrilateral_h
#define NFKQuadrilateral_h

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKQuadrilateral
	@abstract   Four corners in the image, for a shape an axis-aligned box cannot hold.
	@discussion A page photographed at an angle, a barcode on a curved label, and a rectangle seen in
				perspective are all quadrilaterals. A bounding box around one includes what is not
				part of it, and loses the rotation a caller needs to unwarp it.

				Corners are normalized to 0...1 in the input image, origin top-left, the same space
				NFKDetection and NFKKeypoint use, and are named by their place in the shape rather
				than their order in the array. An engine reports them in whatever order it finds
				them; the names are what a caller reads.

				NFKDetection carries one for the engines that report corners, so a barcode is a
				detection with a payload, a box, and the corners of the code itself.
				Introduced in InferKit 0.4.0.
*/
@interface NFKQuadrilateral : NSObject <NSCopying, NSSecureCoding>

/*! The corner at the top left of the shape. */
@property (nonatomic, readonly) CGPoint topLeft;

/*! The corner at the top right of the shape. */
@property (nonatomic, readonly) CGPoint topRight;

/*! The corner at the bottom left of the shape. */
@property (nonatomic, readonly) CGPoint bottomLeft;

/*! The corner at the bottom right of the shape. */
@property (nonatomic, readonly) CGPoint bottomRight;

/*! The smallest axis-aligned box holding all four corners. */
@property (nonatomic, readonly) CGRect boundingBox;

+ (instancetype)quadrilateralWithTopLeft:(CGPoint)topLeft
								topRight:(CGPoint)topRight
							  bottomLeft:(CGPoint)bottomLeft
							 bottomRight:(CGPoint)bottomRight;

- (instancetype)initWithTopLeft:(CGPoint)topLeft
					   topRight:(CGPoint)topRight
					 bottomLeft:(CGPoint)bottomLeft
					bottomRight:(CGPoint)bottomRight NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKQuadrilateral_h */
