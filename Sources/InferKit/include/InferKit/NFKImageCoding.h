//
//  NFKImageCoding.h
//  InferKit
//

#ifndef NFKImageCoding_h
#define NFKImageCoding_h

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreVideo/CoreVideo.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKImageCoding
	@abstract   Encoded image bytes to and from the image representations the contract carries.
	@discussion A remote service takes an image as PNG bytes and returns one the same way, while
				a request carries a CGImage, a CVPixelBuffer, or an MTLTexture under NFKInputImage
				and a backend answers with a CVPixelBuffer under NFKOutputImage. This is the
				conversion between the two, over ImageIO. The three representations are accepted
				wherever an image is taken; a texture must be BGRA8 or RGBA8, a pixel buffer 32BGRA
				or 32RGBA, and anything else answers nil rather than a wrong picture. Decoding
				accepts any format ImageIO reads. Introduced in InferKit 0.3.0.
*/
@interface NFKImageCoding : NSObject

- (instancetype)init NS_UNAVAILABLE;

/*! PNG bytes for a CGImage, CVPixelBuffer, or MTLTexture, or nil where the representation is not one of those. */
+ (nullable NSData *)PNGDataForImage:(id)image;

/*! A data URL ("data:image/png;base64,…") for an image, which is how a chat endpoint takes one inline. */
+ (nullable NSString *)dataURLForImage:(id)image;

/*! A CGImage for a CGImage, CVPixelBuffer, or MTLTexture. The caller releases it. */
+ (nullable CGImageRef)CGImageForImage:(id)image CF_RETURNS_RETAINED;

/*! A 32BGRA pixel buffer decoded from PNG, JPEG, or any other format ImageIO reads. The caller releases it. */
+ (nullable CVPixelBufferRef)pixelBufferWithImageData:(NSData *)data CF_RETURNS_RETAINED;

/*! A 32BGRA pixel buffer drawn from a CGImage. The caller releases it. */
+ (nullable CVPixelBufferRef)pixelBufferWithCGImage:(CGImageRef)image CF_RETURNS_RETAINED;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKImageCoding_h */
