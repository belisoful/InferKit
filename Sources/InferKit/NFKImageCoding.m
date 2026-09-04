//
//  NFKImageCoding.m
//  InferKit
//

#import <InferKit/NFKImageCoding.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>

@interface NFKImageCoding ()
+ (nullable CGImageRef)CGImageFromPixelBuffer:(CVPixelBufferRef)pixelBuffer CF_RETURNS_RETAINED;
+ (nullable CGImageRef)CGImageFromTexture:(id<MTLTexture>)texture CF_RETURNS_RETAINED;
+ (nullable CGImageRef)CGImageWithBytes:(NSData *)bytes
								  width:(size_t)width
								 height:(size_t)height
							bytesPerRow:(size_t)bytesPerRow
							 bitmapInfo:(CGBitmapInfo)bitmapInfo CF_RETURNS_RETAINED;
@end

@implementation NFKImageCoding

#pragma mark Encoding

+ (nullable NSData *)PNGDataForImage:(id)image
{
	CGImageRef cgImage = [self CGImageForImage:image];
	if (cgImage == NULL) {
		return nil;
	}
	NSMutableData *data = [NSMutableData data];
	CGImageDestinationRef destination = CGImageDestinationCreateWithData((__bridge CFMutableDataRef)data,
																		 (__bridge CFStringRef)@"public.png", 1, NULL);
	if (destination == NULL) {
		CGImageRelease(cgImage);
		return nil;
	}
	CGImageDestinationAddImage(destination, cgImage, NULL);
	BOOL written = CGImageDestinationFinalize(destination);
	CFRelease(destination);
	CGImageRelease(cgImage);
	return written ? data : nil;
}

+ (nullable NSString *)dataURLForImage:(id)image
{
	NSData *png = [self PNGDataForImage:image];
	if (png == nil) {
		return nil;
	}
	return [@"data:image/png;base64," stringByAppendingString:[png base64EncodedStringWithOptions:0]];
}

+ (nullable CGImageRef)CGImageForImage:(id)image
{
	if (image == nil) {
		return NULL;
	}
	if ([image conformsToProtocol:@protocol(MTLTexture)]) {
		return [self CGImageFromTexture:(id<MTLTexture>)image];
	}
	CFTypeID type = CFGetTypeID((__bridge CFTypeRef)image);
	if (type == CGImageGetTypeID()) {
		return CGImageRetain((__bridge CGImageRef)image);
	}
	if (type == CVPixelBufferGetTypeID()) {
		return [self CGImageFromPixelBuffer:(__bridge CVPixelBufferRef)image];
	}
	return NULL;
}

// The pixel buffer is copied out under its lock, since the CGImage may outlive the buffer.
+ (nullable CGImageRef)CGImageFromPixelBuffer:(CVPixelBufferRef)pixelBuffer
{
	OSType format = CVPixelBufferGetPixelFormatType(pixelBuffer);
	CGBitmapInfo bitmapInfo;
	if (format == kCVPixelFormatType_32BGRA) {
		bitmapInfo = kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst;
	} else if (format == kCVPixelFormatType_32RGBA) {
		bitmapInfo = kCGBitmapByteOrder32Big | kCGImageAlphaPremultipliedLast;
	} else {
		return NULL;
	}
	if (CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess) {
		return NULL;
	}
	size_t width = CVPixelBufferGetWidth(pixelBuffer);
	size_t height = CVPixelBufferGetHeight(pixelBuffer);
	size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);
	NSData *bytes = [NSData dataWithBytes:CVPixelBufferGetBaseAddress(pixelBuffer) length:bytesPerRow * height];
	CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
	return [self CGImageWithBytes:bytes width:width height:height bytesPerRow:bytesPerRow bitmapInfo:bitmapInfo];
}

+ (nullable CGImageRef)CGImageFromTexture:(id<MTLTexture>)texture
{
	CGBitmapInfo bitmapInfo;
	switch (texture.pixelFormat) {
		case MTLPixelFormatBGRA8Unorm:
		case MTLPixelFormatBGRA8Unorm_sRGB:
			bitmapInfo = kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst;
			break;
		case MTLPixelFormatRGBA8Unorm:
		case MTLPixelFormatRGBA8Unorm_sRGB:
			bitmapInfo = kCGBitmapByteOrder32Big | kCGImageAlphaPremultipliedLast;
			break;
		default:
			return NULL;
	}
	size_t width = texture.width;
	size_t height = texture.height;
	size_t bytesPerRow = width * 4;
	NSMutableData *bytes = [NSMutableData dataWithLength:bytesPerRow * height];
	[texture getBytes:bytes.mutableBytes bytesPerRow:bytesPerRow
		   fromRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0];
	return [self CGImageWithBytes:bytes width:width height:height bytesPerRow:bytesPerRow bitmapInfo:bitmapInfo];
}

+ (nullable CGImageRef)CGImageWithBytes:(NSData *)bytes
								  width:(size_t)width
								 height:(size_t)height
							bytesPerRow:(size_t)bytesPerRow
							 bitmapInfo:(CGBitmapInfo)bitmapInfo
{
	CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)bytes);
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGImageRef image = CGImageCreate(width, height, 8, 32, bytesPerRow, colorSpace, bitmapInfo,
									 provider, NULL, false, kCGRenderingIntentDefault);
	CGColorSpaceRelease(colorSpace);
	CGDataProviderRelease(provider);
	return image;
}

#pragma mark Decoding

+ (nullable CVPixelBufferRef)pixelBufferWithImageData:(NSData *)data
{
	CGImageSourceRef source = CGImageSourceCreateWithData((__bridge CFDataRef)data, NULL);
	if (source == NULL) {
		return NULL;
	}
	CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
	CFRelease(source);
	if (image == NULL) {
		return NULL;
	}
	CVPixelBufferRef pixelBuffer = [self pixelBufferWithCGImage:image];
	CGImageRelease(image);
	return pixelBuffer;
}

+ (nullable CVPixelBufferRef)pixelBufferWithCGImage:(CGImageRef)image
{
	size_t width = CGImageGetWidth(image);
	size_t height = CGImageGetHeight(image);
	NSDictionary *attributes = @{ (__bridge NSString *)kCVPixelBufferCGImageCompatibilityKey: @YES,
								  (__bridge NSString *)kCVPixelBufferCGBitmapContextCompatibilityKey: @YES,
								  (__bridge NSString *)kCVPixelBufferIOSurfacePropertiesKey: @{} };
	CVPixelBufferRef pixelBuffer = NULL;
	if (CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
							(__bridge CFDictionaryRef)attributes, &pixelBuffer) != kCVReturnSuccess) {
		return NULL;
	}
	CVPixelBufferLockBaseAddress(pixelBuffer, 0);
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(CVPixelBufferGetBaseAddress(pixelBuffer), width, height, 8,
												 CVPixelBufferGetBytesPerRow(pixelBuffer), colorSpace,
												 kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	if (context != NULL) {
		CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
		CGContextRelease(context);
	}
	CGColorSpaceRelease(colorSpace);
	CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
	if (context == NULL) {
		CVPixelBufferRelease(pixelBuffer);
		return NULL;
	}
	return pixelBuffer;
}

@end
