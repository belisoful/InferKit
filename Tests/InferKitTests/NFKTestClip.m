//
//  NFKTestClip.m
//  InferKitTests
//

#import "NFKTestClip.h"
#import <AVFoundation/AVFoundation.h>

@implementation NFKTestClip

+ (nullable NSURL *)writeClipWithColors:(NSArray<NSArray<NSNumber *> *> *)colors error:(NSError * _Nullable *)outError
{
	NSURL *url = [[NSURL fileURLWithPath:NSTemporaryDirectory()]
				  URLByAppendingPathComponent:[NSString stringWithFormat:@"inferkit-clip-%@.mp4", NSUUID.UUID.UUIDString]];
	AVAssetWriter *writer = [AVAssetWriter assetWriterWithURL:url fileType:AVFileTypeMPEG4 error:outError];
	if (writer == nil) {
		return nil;
	}
	NSDictionary *settings = @{ AVVideoCodecKey: AVVideoCodecTypeH264, AVVideoWidthKey: @64, AVVideoHeightKey: @64 };
	AVAssetWriterInput *input = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:settings];
	input.expectsMediaDataInRealTime = NO;
	AVAssetWriterInputPixelBufferAdaptor *adaptor =
		[AVAssetWriterInputPixelBufferAdaptor assetWriterInputPixelBufferAdaptorWithAssetWriterInput:input
															   sourcePixelBufferAttributes:@{ (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
																							  (id)kCVPixelBufferWidthKey: @64,
																							  (id)kCVPixelBufferHeightKey: @64 }];
	[writer addInput:input];
	if (![writer startWriting]) {
		if (outError != NULL) { *outError = writer.error; }
		return nil;
	}
	[writer startSessionAtSourceTime:kCMTimeZero];

	// The pool hands a released buffer straight back for the next frame while the encoder may still
	// be reading it, so every buffer is held until the writer has finished.
	NSMutableArray *held = [NSMutableArray array];
	for (NSUInteger index = 0; index < colors.count; index++) {
		while (!input.readyForMoreMediaData) {
			[NSThread sleepForTimeInterval:0.01];
		}
		CVPixelBufferRef buffer = NULL;
		CVPixelBufferPoolCreatePixelBuffer(NULL, adaptor.pixelBufferPool, &buffer);
		if (buffer == NULL) {
			break;
		}
		CVPixelBufferLockBaseAddress(buffer, 0);
		uint8_t *base = CVPixelBufferGetBaseAddress(buffer);
		size_t bytesPerRow = CVPixelBufferGetBytesPerRow(buffer);
		NSArray<NSNumber *> *rgb = colors[index];
		for (size_t y = 0; y < 64; y++) {
			for (size_t x = 0; x < 64; x++) {
				uint8_t *pixel = base + y * bytesPerRow + x * 4;
				pixel[0] = (uint8_t)(rgb[2].doubleValue * 255);
				pixel[1] = (uint8_t)(rgb[1].doubleValue * 255);
				pixel[2] = (uint8_t)(rgb[0].doubleValue * 255);
				pixel[3] = 255;
			}
		}
		CVPixelBufferUnlockBaseAddress(buffer, 0);
		[adaptor appendPixelBuffer:buffer withPresentationTime:CMTimeMake((int64_t)index, 2)];
		[held addObject:(__bridge id)buffer];
		CVPixelBufferRelease(buffer);
	}
	[input markAsFinished];
	// The session ends one frame past the last presentation time so the last frame has a duration.
	[writer endSessionAtSourceTime:CMTimeMake((int64_t)colors.count, 2)];

	dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
	[writer finishWritingWithCompletionHandler:^{ dispatch_semaphore_signal(semaphore); }];
	dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
	[held removeAllObjects];
	if (writer.status != AVAssetWriterStatusCompleted) {
		if (outError != NULL) { *outError = writer.error; }
		return nil;
	}
	return url;
}

+ (NSArray<NSNumber *> *)meanColorOfImage:(CGImageRef)image
{
	size_t width = CGImageGetWidth(image), height = CGImageGetHeight(image);
	NSMutableData *bytes = [NSMutableData dataWithLength:width * height * 4];
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(bytes.mutableBytes, width, height, 8, width * 4, colorSpace,
												 kCGBitmapByteOrder32Big | kCGImageAlphaPremultipliedLast);
	CGColorSpaceRelease(colorSpace);
	CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
	CGContextRelease(context);
	const uint8_t *pixels = bytes.bytes;
	double r = 0, g = 0, b = 0;
	for (size_t i = 0; i < width * height; i++) {
		r += pixels[i * 4];
		g += pixels[i * 4 + 1];
		b += pixels[i * 4 + 2];
	}
	double count = (double)(width * height) * 255.0;
	return @[ @(r / count), @(g / count), @(b / count) ];
}

@end
