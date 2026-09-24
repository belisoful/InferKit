//
//  NFKVisionRectangleBackend.h
//  InferKit
//

#ifndef NFKVisionRectangleBackend_h
#define NFKVisionRectangleBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@enum       NFKVisionRectangleKind
	@abstract   Which four-cornered shape the backend looks for.
	@constant   NFKVisionRectangleKindRectangle  Any rectangle, filtered by the properties below.
	@constant   NFKVisionRectangleKindDocument  The page in a photograph (macOS 12, iOS 15, tvOS 15).
	@constant   NFKVisionRectangleKindBarcode  A barcode or QR code, read as well as located.
*/
typedef NS_ENUM(NSInteger, NFKVisionRectangleKind) {
	NFKVisionRectangleKindRectangle	= 0,
	NFKVisionRectangleKindDocument	= 1,
	NFKVisionRectangleKindBarcode	= 2,
};

/*!
	@class      NFKVisionRectangleBackend
	@abstract   Finds four-cornered shapes in an image through Apple's Vision framework.
	@discussion NFKInputImage in, one NFKDetection per shape under NFKOutputDetections, each carrying
				a `quadrilateral` because a page or a code seen at an angle is not axis-aligned. The
				bounding box is the corners' own, so a caller that wants a box still has one.

				A barcode's detection is labeled with its payload, which is the string the code
				carries; a code whose payload is not text has no label. `symbologies` narrows the
				search on the way in, which is where that choice belongs, and reading it back out of
				a result is not part of the contract. `supportedSymbologies` reports what the
				installed Vision revision can read.

				The rectangle kind is geometric rather than semantic: it finds shapes, and the
				properties below say which. Introduced in InferKit 0.4.0.
*/
@interface NFKVisionRectangleBackend : NSObject <NFKInferenceBackend>

/*! What to look for. Any rectangle by default. */
@property (nonatomic) NFKVisionRectangleKind kind;

/*! The most shapes to report. 16 by default, and 1 for the document kind, which finds one page. */
@property (nonatomic) NSInteger maximumCount;

/*! The smallest side, as a fraction of the image. 0.2 by default. The rectangle kind reads it. */
@property (nonatomic) double minimumSize;

/*! How far from square a corner may be, in degrees. 30 by default. The rectangle kind reads it. */
@property (nonatomic) double quadratureTolerance;

/*! The barcode symbologies to look for (for example @[@"VNBarcodeSymbologyQR"]). nil looks for
	every symbology the revision supports. */
@property (nonatomic, copy, nullable) NSArray<NSString *> *symbologies;

/*! The symbologies the installed Vision revision reads. */
@property (class, nonatomic, readonly, copy) NSArray<NSString *> *supportedSymbologies;

+ (instancetype)backend;

/*! A backend for one kind. */
+ (instancetype)backendWithKind:(NFKVisionRectangleKind)kind;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKVisionRectangleBackend_h */
