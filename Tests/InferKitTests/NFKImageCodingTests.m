//
//  NFKImageCodingTests.m
//  InferKitTests
//
//  The PNG round trip through each representation the contract carries. A drawn CGImage with known
//  pixels is the source; what comes back through the codec must hold the same pixels.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKImageCoding.h>
#import <Metal/Metal.h>

@interface NFKImageCodingTests : XCTestCase
@end

@implementation NFKImageCodingTests

// A 4×3 image: red, green, blue, white across the top row, black elsewhere.
- (CGImageRef)makeTestImage CF_RETURNS_RETAINED
{
	size_t width = 4, height = 3;
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, width, height, 8, width * 4, colorSpace,
												 kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	CGContextSetRGBFillColor(context, 0, 0, 0, 1);
	CGContextFillRect(context, CGRectMake(0, 0, width, height));
	CGFloat colors[4][3] = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1} };
	for (size_t x = 0; x < width; x++) {
		CGContextSetRGBFillColor(context, colors[x][0], colors[x][1], colors[x][2], 1);
		// CoreGraphics is y-up: the top row is y == height - 1.
		CGContextFillRect(context, CGRectMake(x, height - 1, 1, 1));
	}
	CGImageRef image = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	return image;
}

- (NSArray<NSNumber *> *)topRowOfPixelBuffer:(CVPixelBufferRef)pixelBuffer
{
	CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
	const uint8_t *row = CVPixelBufferGetBaseAddress(pixelBuffer);   // row 0 is the top in a pixel buffer
	NSMutableArray<NSNumber *> *bgra = [NSMutableArray array];
	for (size_t i = 0; i < CVPixelBufferGetWidth(pixelBuffer) * 4; i++) {
		[bgra addObject:@(row[i])];
	}
	CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
	return bgra;
}

- (NSArray<NSNumber *> *)expectedTopRowBGRA
{
	return @[ @0, @0, @255, @255,   @0, @255, @0, @255,   @255, @0, @0, @255,   @255, @255, @255, @255 ];
}

- (void)testACGImageRoundTripsThroughPNGToAPixelBuffer
{
	CGImageRef source = [self makeTestImage];
	NSData *png = [NFKImageCoding PNGDataForImage:(__bridge id)source];
	XCTAssertNotNil(png);
	const uint8_t *signature = png.bytes;
	XCTAssertEqual(signature[1], 'P');
	XCTAssertEqual(signature[2], 'N');
	XCTAssertEqual(signature[3], 'G');

	CVPixelBufferRef decoded = [NFKImageCoding pixelBufferWithImageData:png];
	XCTAssertTrue(decoded != NULL);
	XCTAssertEqual(CVPixelBufferGetPixelFormatType(decoded), kCVPixelFormatType_32BGRA);
	XCTAssertEqual(CVPixelBufferGetWidth(decoded), 4);
	XCTAssertEqual(CVPixelBufferGetHeight(decoded), 3);
	XCTAssertEqualObjects([self topRowOfPixelBuffer:decoded], [self expectedTopRowBGRA]);
	CVPixelBufferRelease(decoded);
	CGImageRelease(source);
}

- (void)testAPixelBufferEncodesToTheSamePNGPixels
{
	CGImageRef source = [self makeTestImage];
	CVPixelBufferRef pixelBuffer = [NFKImageCoding pixelBufferWithCGImage:source];
	XCTAssertTrue(pixelBuffer != NULL);
	XCTAssertEqualObjects([self topRowOfPixelBuffer:pixelBuffer], [self expectedTopRowBGRA]);

	NSData *png = [NFKImageCoding PNGDataForImage:(__bridge id)pixelBuffer];
	XCTAssertNotNil(png);
	CVPixelBufferRef again = [NFKImageCoding pixelBufferWithImageData:png];
	XCTAssertEqualObjects([self topRowOfPixelBuffer:again], [self expectedTopRowBGRA]);
	CVPixelBufferRelease(again);
	CVPixelBufferRelease(pixelBuffer);
	CGImageRelease(source);
}

- (void)testATextureEncodesToTheSamePNGPixels
{
	id<MTLDevice> device = MTLCreateSystemDefaultDevice();
	if (device == nil) {
		XCTSkip("no Metal device");
	}
	CGImageRef source = [self makeTestImage];
	CVPixelBufferRef pixelBuffer = [NFKImageCoding pixelBufferWithCGImage:source];
	MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
																						  width:4 height:3 mipmapped:NO];
	id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
	CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
	[texture replaceRegion:MTLRegionMake2D(0, 0, 4, 3) mipmapLevel:0
				 withBytes:CVPixelBufferGetBaseAddress(pixelBuffer) bytesPerRow:CVPixelBufferGetBytesPerRow(pixelBuffer)];
	CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

	NSData *png = [NFKImageCoding PNGDataForImage:texture];
	XCTAssertNotNil(png);
	CVPixelBufferRef decoded = [NFKImageCoding pixelBufferWithImageData:png];
	XCTAssertEqualObjects([self topRowOfPixelBuffer:decoded], [self expectedTopRowBGRA]);
	CVPixelBufferRelease(decoded);
	CVPixelBufferRelease(pixelBuffer);
	CGImageRelease(source);
}

- (void)testTheDataURLIsBase64PNG
{
	CGImageRef source = [self makeTestImage];
	NSString *url = [NFKImageCoding dataURLForImage:(__bridge id)source];
	XCTAssertTrue([url hasPrefix:@"data:image/png;base64,"]);
	NSData *decoded = [[NSData alloc] initWithBase64EncodedString:[url substringFromIndex:@"data:image/png;base64,".length] options:0];
	XCTAssertEqualObjects(decoded, [NFKImageCoding PNGDataForImage:(__bridge id)source]);
	CGImageRelease(source);
}

- (void)testAnUnsupportedRepresentationAnswersNilRatherThanAWrongPicture
{
	XCTAssertNil([NFKImageCoding PNGDataForImage:@"not an image"]);
	XCTAssertNil([NFKImageCoding dataURLForImage:@[]]);
	XCTAssertTrue([NFKImageCoding pixelBufferWithImageData:[@"not a png" dataUsingEncoding:NSUTF8StringEncoding]] == NULL);

	CVPixelBufferRef gray = NULL;
	CVPixelBufferCreate(kCFAllocatorDefault, 2, 2, kCVPixelFormatType_OneComponent8, NULL, &gray);
	XCTAssertNil([NFKImageCoding PNGDataForImage:(__bridge id)gray], @"a one-channel buffer is not a supported layout");
	CVPixelBufferRelease(gray);
}

@end
