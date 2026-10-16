//
//  NFKWireAttachmentTests.m
//  InferKitTests
//
//  The wire coding's attachment mode, which a plugin channel uses: every attachment kind, padded planes,
//  the inline threshold, files by reference, an IOSurface buffer copied into shared memory, a request
//  carried by a real frame, and the HTTP form left as it was.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

#if TARGET_OS_OSX

#import "../../Sources/InferKit/NFKInferenceWireCoding.h"
#import "../../Sources/InferKit/NFKPluginChannel.h"
#import "../../Sources/InferKit/NFKPluginSharedMemory.h"
#import "../../Sources/InferKit/NFKPluginWire.h"
#import <AVFoundation/AVFoundation.h>
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <fcntl.h>
#import <mach/vm_page_size.h>
#import <unistd.h>

@interface NFKWireAttachmentTests : XCTestCase
@property (nonatomic, strong) NSURL *directory;
@end

@implementation NFKWireAttachmentTests

- (void)setUp
{
	NSString *name = [NSString stringWithFormat:@"wire-attachments-%@", NSUUID.UUID.UUIDString];
	self.directory = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:name isDirectory:YES];
	[NSFileManager.defaultManager createDirectoryAtURL:self.directory withIntermediateDirectories:YES attributes:nil error:NULL];
}

- (void)tearDown
{
	[NSFileManager.defaultManager removeItemAtURL:self.directory error:NULL];
}

#pragma mark Helpers

- (NFKInferenceWireCoding *)encoderWithWriter:(NFKPluginAttachmentWriter *)writer
{
	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:self.directory];
	coder.attachmentSink = writer;
	return coder;
}

/*! The descriptors a frame would deliver: the kernel gives the receiver duplicates. */
- (NFKPluginAttachmentReader *)readerForWriter:(NFKPluginAttachmentWriter *)writer
{
	NSMutableArray<NFKPluginDescriptor *> *received = [NSMutableArray array];
	for (NSNumber *descriptor in writer.descriptors) {
		[received addObject:[[NFKPluginDescriptor alloc] initWithDescriptor:dup(descriptor.intValue)]];
	}
	return [[NFKPluginAttachmentReader alloc] initWithDescriptors:received];
}

/*! Encodes value with an attachment writer, carries the entry through JSON text, and decodes it with a
	reader over the writer's descriptors. */
- (nullable id)roundTrip:(id)value entry:(NSDictionary * _Nullable * _Nullable)outEntry writer:(NFKPluginAttachmentWriter * _Nullable * _Nullable)outWriter
{
	NFKPluginAttachmentWriter *writer NS_VALID_UNTIL_END_OF_SCOPE = [[NFKPluginAttachmentWriter alloc] init];
	NSError *error = nil;
	id entry = [[self encoderWithWriter:writer] JSONValueForValue:value key:@"value" error:&error];
	XCTAssertNotNil(entry, @"%@", error);
	NSData *text = [NSJSONSerialization dataWithJSONObject:@[ entry ?: @{} ] options:0 error:&error];
	XCTAssertNotNil(text, @"the entry is plain JSON: %@", error);
	entry = [[NSJSONSerialization JSONObjectWithData:text options:0 error:NULL] firstObject];
	if (outEntry != NULL) {
		*outEntry = entry;
	}
	if (outWriter != NULL) {
		*outWriter = writer;
	}
	NFKInferenceWireCoding *decoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:self.directory];
	NFKPluginAttachmentReader *reader = [self readerForWriter:writer];
	reader.temporaryDirectoryURL = self.directory;
	decoder.attachmentSource = reader;
	id decoded = [decoder valueFromJSONValue:entry key:@"value" error:&error];
	XCTAssertNotNil(decoded, @"%@", error);
	return decoded;
}

- (NSData *)randomDataOfLength:(NSUInteger)length
{
	NSMutableData *data = [NSMutableData dataWithLength:length];
	arc4random_buf(data.mutableBytes, length);
	return data;
}

- (CVPixelBufferRef)newPixelBufferWithWidth:(size_t)width height:(size_t)height format:(OSType)format attributes:(NSDictionary *)attributes
{
	CVPixelBufferRef buffer = NULL;
	CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, (__bridge CFDictionaryRef)attributes, &buffer);
	XCTAssertEqual(status, kCVReturnSuccess);
	CVPixelBufferLockBaseAddress(buffer, 0);
	size_t planes = CVPixelBufferIsPlanar(buffer) ? CVPixelBufferGetPlaneCount(buffer) : 1;
	for (size_t plane = 0; plane < planes; plane++) {
		void *base = CVPixelBufferIsPlanar(buffer) ? CVPixelBufferGetBaseAddressOfPlane(buffer, plane) : CVPixelBufferGetBaseAddress(buffer);
		size_t rows = CVPixelBufferIsPlanar(buffer) ? CVPixelBufferGetHeightOfPlane(buffer, plane) : CVPixelBufferGetHeight(buffer);
		size_t bytesPerRow = CVPixelBufferIsPlanar(buffer) ? CVPixelBufferGetBytesPerRowOfPlane(buffer, plane) : CVPixelBufferGetBytesPerRow(buffer);
		arc4random_buf(base, rows * bytesPerRow);
	}
	CVPixelBufferUnlockBaseAddress(buffer, 0);
	return buffer;
}

/*! Compares the bytes each row's pixels occupy in every plane, ignoring row padding. */
- (void)assertPixelsOf:(nullable CVPixelBufferRef)decoded equal:(CVPixelBufferRef)source
{
	// A failed decode reads no planes: their base addresses would be NULL.
	if (decoded == NULL || CFGetTypeID(decoded) != CVPixelBufferGetTypeID()) {
		XCTFail(@"no pixel buffer was decoded");
		return;
	}
	XCTAssertEqual(CVPixelBufferGetWidth(decoded), CVPixelBufferGetWidth(source));
	XCTAssertEqual(CVPixelBufferGetHeight(decoded), CVPixelBufferGetHeight(source));
	XCTAssertEqual(CVPixelBufferGetPixelFormatType(decoded), CVPixelBufferGetPixelFormatType(source));
	CVPixelBufferLockBaseAddress(source, kCVPixelBufferLock_ReadOnly);
	CVPixelBufferLockBaseAddress(decoded, kCVPixelBufferLock_ReadOnly);
	BOOL planar = CVPixelBufferIsPlanar(source);
	size_t planes = planar ? CVPixelBufferGetPlaneCount(source) : 1;
	XCTAssertEqual(planar ? CVPixelBufferGetPlaneCount(decoded) : 1, planes);
	for (size_t plane = 0; plane < planes; plane++) {
		const uint8_t *left = planar ? CVPixelBufferGetBaseAddressOfPlane(decoded, plane) : CVPixelBufferGetBaseAddress(decoded);
		const uint8_t *right = planar ? CVPixelBufferGetBaseAddressOfPlane(source, plane) : CVPixelBufferGetBaseAddress(source);
		size_t leftStride = planar ? CVPixelBufferGetBytesPerRowOfPlane(decoded, plane) : CVPixelBufferGetBytesPerRow(decoded);
		size_t rightStride = planar ? CVPixelBufferGetBytesPerRowOfPlane(source, plane) : CVPixelBufferGetBytesPerRow(source);
		size_t rows = planar ? CVPixelBufferGetHeightOfPlane(source, plane) : CVPixelBufferGetHeight(source);
		size_t width = planar ? CVPixelBufferGetWidthOfPlane(source, plane) : CVPixelBufferGetWidth(source);
		// 32BGRA holds 4 bytes a pixel; 420v holds 1 in its luma plane and 2 in its chroma plane.
		size_t bytesPerPixel = planar ? (plane == 0 ? 1 : 2) : 4;
		for (size_t row = 0; row < rows; row++) {
			XCTAssertEqual(memcmp(left + row * leftStride, right + row * rightStride, width * bytesPerPixel), 0,
						   @"plane %zu row %zu", plane, row);
		}
	}
	CVPixelBufferUnlockBaseAddress(decoded, kCVPixelBufferLock_ReadOnly);
	CVPixelBufferUnlockBaseAddress(source, kCVPixelBufferLock_ReadOnly);
}

- (CGImageRef)newNoiseImageWithSide:(size_t)side
{
	NSData *pixels = [self randomDataOfLength:side * side * 4];
	CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate((void *)pixels.bytes, side, side, 8, side * 4, space,
												 (CGBitmapInfo)kCGImageAlphaNoneSkipLast);
	CGImageRef image = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	CGColorSpaceRelease(space);
	return image;
}

#pragma mark Kinds

- (void)testDataAboveTheLimitCrossesAsAnAttachment
{
	NSData *data = [self randomDataOfLength:100 * 1024];
	NSDictionary *entry = nil;
	NSData *decoded = [self roundTrip:data entry:&entry writer:NULL];
	XCTAssertEqualObjects(entry[@"$nfk"], @"buffer");
	XCTAssertEqualObjects(entry[@"as"], @"data");
	XCTAssertEqualObjects(entry[@"index"], @0);
	XCTAssertEqualObjects(entry[@"length"], @(data.length));
	XCTAssertNil(entry[@"base64"], @"the bytes leave the JSON");
	XCTAssertEqualObjects(decoded, data);
}

- (void)testFloatingArraysCrossAsPackedAttachments
{
	NSMutableArray<NSNumber *> *singles = [NSMutableArray array];
	NSMutableArray<NSNumber *> *doubles = [NSMutableArray array];
	for (NSUInteger index = 0; index < 20000; index++) {
		[singles addObject:@((float)index * 0.25f + 0.1f)];
		[doubles addObject:@((double)index / 3.0)];
	}
	NSDictionary *entry = nil;
	NSArray *decoded = [self roundTrip:singles entry:&entry writer:NULL];
	XCTAssertEqualObjects(entry[@"as"], @"float32Array");
	XCTAssertEqualObjects(entry[@"count"], @20000);
	XCTAssertEqualObjects(decoded, singles);
	decoded = [self roundTrip:doubles entry:&entry writer:NULL];
	XCTAssertEqualObjects(entry[@"as"], @"float64Array");
	XCTAssertEqualObjects(decoded, doubles, @"doubles cross bit for bit");
}

- (void)testAMultiArrayWrapsItsAttachmentWithoutACopy
{
	NSError *error = nil;
	MLMultiArray *array = [[MLMultiArray alloc] initWithShape:@[ @64, @512 ] dataType:MLMultiArrayDataTypeFloat32 error:&error];
	XCTAssertNotNil(array, @"%@", error);
	float *values = array.dataPointer;
	for (NSInteger index = 0; index < array.count; index++) {
		values[index] = (float)index * 0.5f;
	}
	NSDictionary *entry = nil;
	NFKPluginAttachmentWriter *writer = nil;
	MLMultiArray *decoded = [self roundTrip:array entry:&entry writer:&writer];
	if (![decoded isKindOfClass:MLMultiArray.class]) {
		return;
	}
	XCTAssertEqualObjects(entry[@"as"], @"multiArray");
	XCTAssertEqualObjects(entry[@"shape"], (@[ @64, @512 ]));
	XCTAssertEqualObjects(entry[@"dataType"], @(MLMultiArrayDataTypeFloat32));
	XCTAssertTrue([decoded isKindOfClass:MLMultiArray.class]);
	XCTAssertEqualObjects(decoded.shape, array.shape);
	XCTAssertEqual(memcmp(decoded.dataPointer, array.dataPointer, (size_t)array.count * sizeof(float)), 0);

	// The decoded array reads the shared region itself: a write through another mapping shows in it.
	NFKPluginSharedMemory *region = [NFKPluginSharedMemory regionWithDescriptor:dup(writer.descriptors.firstObject.intValue) copiesFiles:YES error:&error];
	XCTAssertNotNil(region, @"%@", error);
	((float *)region.mutableBytes)[3] = 1234.5f;
	XCTAssertEqual(((const float *)decoded.dataPointer)[3], 1234.5f);
}

- (void)testAPixelBufferKeepsItsPaddedRows
{
	NSDictionary *attributes = @{ (id)kCVPixelBufferBytesPerRowAlignmentKey: @256 };
	CVPixelBufferRef source = [self newPixelBufferWithWidth:250 height:96 format:kCVPixelFormatType_32BGRA attributes:attributes];
	size_t stride = CVPixelBufferGetBytesPerRow(source);
	XCTAssertGreaterThan(stride, (size_t)250 * 4, @"the rows carry padding");
	NSDictionary *entry = nil;
	id decoded = [self roundTrip:(__bridge id)source entry:&entry writer:NULL];
	XCTAssertEqualObjects(entry[@"as"], @"pixelBuffer");
	XCTAssertEqualObjects(entry[@"format"], @(kCVPixelFormatType_32BGRA));
	NSDictionary *plane = [entry[@"planes"] firstObject];
	XCTAssertEqualObjects(plane[@"offset"], @0);
	XCTAssertEqualObjects(plane[@"bytesPerRow"], @(stride), @"the padding crosses as the buffer holds it");
	XCTAssertEqualObjects(plane[@"rows"], @96);
	XCTAssertTrue(decoded != nil && CFGetTypeID((__bridge CFTypeRef)decoded) == CVPixelBufferGetTypeID());
	if (decoded != nil) {
		XCTAssertEqual(CVPixelBufferGetBytesPerRow((__bridge CVPixelBufferRef)decoded), stride, @"the wrapped buffer keeps the stride");
	}
	[self assertPixelsOf:(__bridge CVPixelBufferRef)decoded equal:source];
	CVPixelBufferRelease(source);
}

- (void)testABiplanarPixelBufferCrossesWithAlignedPlanes
{
	CVPixelBufferRef source = [self newPixelBufferWithWidth:300 height:200 format:kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange attributes:@{}];
	NSDictionary *entry = nil;
	id decoded = [self roundTrip:(__bridge id)source entry:&entry writer:NULL];
	NSArray<NSDictionary *> *planes = entry[@"planes"];
	XCTAssertEqual(planes.count, 2u);
	XCTAssertEqual([planes[1][@"offset"] unsignedLongValue] % 64, 0u, @"each plane starts aligned");
	XCTAssertGreaterThanOrEqual([planes[1][@"offset"] unsignedLongValue],
								[planes[0][@"bytesPerRow"] unsignedLongValue] * [planes[0][@"rows"] unsignedLongValue]);
	XCTAssertTrue(decoded != nil && CVPixelBufferIsPlanar((__bridge CVPixelBufferRef)decoded));
	[self assertPixelsOf:(__bridge CVPixelBufferRef)decoded equal:source];
	CVPixelBufferRelease(source);
}

- (void)testPCMBuffersCrossInBothLayouts
{
	AVAudioFormat *planar = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32 sampleRate:48000 channels:2 interleaved:NO];
	AVAudioPCMBuffer *source = [[AVAudioPCMBuffer alloc] initWithPCMFormat:planar frameCapacity:20000];
	source.frameLength = 20000;
	for (AVAudioFrameCount frame = 0; frame < source.frameLength; frame++) {
		source.floatChannelData[0][frame] = sinf((float)frame * 0.01f);
		source.floatChannelData[1][frame] = cosf((float)frame * 0.01f);
	}
	NSDictionary *entry = nil;
	AVAudioPCMBuffer *decoded = [self roundTrip:source entry:&entry writer:NULL];
	if (![decoded isKindOfClass:AVAudioPCMBuffer.class]) {
		return;
	}
	XCTAssertEqualObjects(entry[@"as"], @"pcm");
	XCTAssertEqual([entry[@"buffers"] count], 2u);
	XCTAssertTrue([decoded isKindOfClass:AVAudioPCMBuffer.class]);
	XCTAssertEqual(decoded.frameLength, 20000u);
	XCTAssertEqual(decoded.format.channelCount, 2u);
	XCTAssertEqual(memcmp(decoded.floatChannelData[1], source.floatChannelData[1], 20000 * sizeof(float)), 0);

	AVAudioFormat *interleaved = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatInt16 sampleRate:44100 channels:2 interleaved:YES];
	AVAudioPCMBuffer *shorts = [[AVAudioPCMBuffer alloc] initWithPCMFormat:interleaved frameCapacity:30000];
	shorts.frameLength = 30000;
	for (AVAudioFrameCount sample = 0; sample < 60000; sample++) {
		shorts.int16ChannelData[0][sample] = (int16_t)(sample * 7);
	}
	decoded = [self roundTrip:shorts entry:&entry writer:NULL];
	if (![decoded isKindOfClass:AVAudioPCMBuffer.class]) {
		return;
	}
	XCTAssertEqual([entry[@"buffers"] count], 1u, @"an interleaved buffer is one buffer");
	XCTAssertTrue(decoded.format.isInterleaved);
	XCTAssertEqual(decoded.frameLength, 30000u);
	XCTAssertEqual(memcmp(decoded.int16ChannelData[0], shorts.int16ChannelData[0], 60000 * sizeof(int16_t)), 0);
}

- (void)testImagesCrossAsPNGAttachments
{
	CGImageRef image = [self newNoiseImageWithSide:256];
	NSDictionary *entry = nil;
	id decoded = [self roundTrip:(__bridge id)image entry:&entry writer:NULL];
	XCTAssertEqualObjects(entry[@"$nfk"], @"buffer");
	XCTAssertEqualObjects(entry[@"as"], @"cgImage");
	XCTAssertTrue(decoded != nil && CFGetTypeID((__bridge CFTypeRef)decoded) == CGImageGetTypeID());
	if (decoded != nil) {
		XCTAssertEqual(CGImageGetWidth((__bridge CGImageRef)decoded), 256u);
	}
	CGImageRelease(image);

	id<MTLDevice> device = MTLCreateSystemDefaultDevice();
	if (device == nil) {
		return;
	}
	MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:256 height:256 mipmapped:NO];
	id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
	NSData *pixels = [self randomDataOfLength:256 * 256 * 4];
	[texture replaceRegion:MTLRegionMake2D(0, 0, 256, 256) mipmapLevel:0 withBytes:pixels.bytes bytesPerRow:256 * 4];
	decoded = [self roundTrip:texture entry:&entry writer:NULL];
	XCTAssertEqualObjects(entry[@"as"], @"image");
	XCTAssertTrue(decoded != nil && CFGetTypeID((__bridge CFTypeRef)decoded) == CVPixelBufferGetTypeID(), @"a texture arrives as a pixel buffer");
	if (decoded != nil) {
		XCTAssertEqual(CVPixelBufferGetWidth((__bridge CVPixelBufferRef)decoded), 256u);
	}
}

#pragma mark Threshold, files, surfaces

- (void)testTheInlineThresholdIsTheLargestInlineValue
{
	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	XCTAssertEqual(coder.inlineByteLimit, 65536u);
	NFKPluginAttachmentWriter *writer NS_VALID_UNTIL_END_OF_SCOPE = [[NFKPluginAttachmentWriter alloc] init];
	coder = [self encoderWithWriter:writer];
	coder.inlineByteLimit = 1024;
	NSError *error = nil;
	NSDictionary *atLimit = [coder JSONValueForValue:[self randomDataOfLength:1024] key:@"value" error:&error];
	XCTAssertEqualObjects(atLimit[@"$nfk"], @"data", @"a value at the limit stays inline");
	XCTAssertNotNil(atLimit[@"base64"]);
	NSDictionary *overLimit = [coder JSONValueForValue:[self randomDataOfLength:1025] key:@"value" error:&error];
	XCTAssertEqualObjects(overLimit[@"$nfk"], @"buffer", @"one byte over attaches");
	XCTAssertEqual(writer.descriptors.count, 1u);
	NSDictionary *small = [coder JSONValueForValue:@[ @1.5f, @2.5f ] key:@"value" error:&error];
	XCTAssertEqualObjects(small[@"$nfk"], @"float32Array");
	XCTAssertNotNil(small[@"base64"]);
}

- (void)testFilesCrossByReference
{
	NSURL *file = [self.directory URLByAppendingPathComponent:@"clip.wav"];
	NSData *contents = [self randomDataOfLength:4096];
	XCTAssertTrue([contents writeToURL:file atomically:YES]);

	NSDictionary *entry = nil;
	NSURL *decoded = [self roundTrip:file entry:&entry writer:NULL];
	XCTAssertEqualObjects(entry[@"$nfk"], @"file");
	XCTAssertEqualObjects(entry[@"name"], @"clip.wav");
	XCTAssertEqualObjects(entry[@"path"], file.URLByStandardizingPath.path);
	XCTAssertNil(entry[@"base64"], @"a file crosses by path, whatever its size");
	XCTAssertNil(entry[@"index"]);
	XCTAssertEqualObjects(decoded.URLByStandardizingPath.path, file.URLByStandardizingPath.path);

	NFKAudioAsset *asset = [NFKAudioAsset audioAssetWithFileURL:file durationSeconds:2.5 sampleRate:48000 channelCount:2];
	NFKAudioAsset *decodedAsset = [self roundTrip:asset entry:&entry writer:NULL];
	XCTAssertEqualObjects(entry[@"$nfk"], @"audio");
	XCTAssertEqualObjects(decodedAsset.fileURL.URLByStandardizingPath.path, file.URLByStandardizingPath.path);
	XCTAssertEqual(decodedAsset.durationSeconds, 2.5);
	XCTAssertEqual(decodedAsset.channelCount, 2);

	NFKPluginAttachmentWriter *writer NS_VALID_UNTIL_END_OF_SCOPE = [[NFKPluginAttachmentWriter alloc] init];
	writer.passesFilesAsDescriptors = YES;
	NSError *error = nil;
	NSDictionary *indexed = [[self encoderWithWriter:writer] JSONValueForValue:file key:@"value" error:&error];
	XCTAssertEqualObjects(indexed[@"index"], @0, @"a sandboxed peer receives an open descriptor");
	NFKPluginAttachmentReader *reader = [self readerForWriter:writer];
	reader.temporaryDirectoryURL = self.directory;
	NFKInferenceWireCoding *decoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:self.directory];
	decoder.attachmentSource = reader;
	NSURL *copied = [decoder valueFromJSONValue:indexed key:@"value" error:&error];
	XCTAssertNotNil(copied, @"%@", error);
	XCTAssertNotEqualObjects(copied.path, file.path);
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:copied], contents);
	XCTAssertEqualObjects(copied.pathExtension, @"wav");

	NFKPluginAttachmentReader *fenced = [[NFKPluginAttachmentReader alloc] initWithDescriptors:@[]];
	fenced.allowedFileDirectoryURL = [self.directory URLByAppendingPathComponent:@"outputs" isDirectory:YES];
	decoder.attachmentSource = fenced;
	NSDictionary *outside = @{ @"$nfk": @"file", @"path": file.path, @"name": @"clip.wav" };
	XCTAssertNil([decoder valueFromJSONValue:outside key:@"value" error:&error],
				 @"a path outside the run's directory is refused");
	XCTAssertTrue([error.localizedDescription containsString:@"outside"], @"%@", error);
}

- (void)testAFileReferenceIsCheckedWithItsLinksResolved
{
	NSFileManager *files = NSFileManager.defaultManager;
	NSURL *outputs = [self.directory URLByAppendingPathComponent:@"outputs" isDirectory:YES];
	XCTAssertTrue([files createDirectoryAtURL:outputs withIntermediateDirectories:YES attributes:nil error:NULL]);
	NSURL *secret = [self.directory URLByAppendingPathComponent:@"secret.txt"];
	XCTAssertTrue([@"outside" writeToURL:secret atomically:YES encoding:NSUTF8StringEncoding error:NULL]);
	NSURL *answer = [outputs URLByAppendingPathComponent:@"answer.txt"];
	XCTAssertTrue([@"inside" writeToURL:answer atomically:YES encoding:NSUTF8StringEncoding error:NULL]);
	NSError *error = nil;
	XCTAssertTrue([files createSymbolicLinkAtURL:[outputs URLByAppendingPathComponent:@"escape.txt"] withDestinationURL:secret error:&error], @"%@", error);
	XCTAssertTrue([files createSymbolicLinkAtURL:[outputs URLByAppendingPathComponent:@"folder"] withDestinationURL:self.directory error:&error], @"%@", error);
	XCTAssertTrue([files createSymbolicLinkAtURL:[outputs URLByAppendingPathComponent:@"alias.txt"] withDestinationURL:answer error:&error], @"%@", error);
	NFKPluginAttachmentReader *fenced = [[NFKPluginAttachmentReader alloc] initWithDescriptors:@[]];
	fenced.allowedFileDirectoryURL = outputs;

	for (NSString *name in @[ @"escape.txt", @"folder/secret.txt" ]) {
		NSString *path = [outputs URLByAppendingPathComponent:name].path;
		XCTAssertNil([fenced fileURLForPath:path error:&error], @"%@ names a file outside the directory through a link", name);
		XCTAssertEqualObjects(error.domain, NFKPluginErrorDomain);
		XCTAssertEqual(error.code, NFKPluginErrorProtocolViolation);
		XCTAssertTrue([error.localizedDescription containsString:@"outside"], @"%@", error);
	}
	XCTAssertNil([fenced fileURLForPath:[outputs URLByAppendingPathComponent:@"missing.txt"].path error:&error], @"a missing file is refused");
	XCTAssertNil([fenced fileURLForPath:@"answer.txt" error:&error], @"a relative path is refused");

	NSURL *aliased = [fenced fileURLForPath:[outputs URLByAppendingPathComponent:@"alias.txt"].path error:&error];
	XCTAssertNotNil(aliased, @"a link that stays inside is accepted: %@", error);
	XCTAssertEqualObjects(aliased.lastPathComponent, @"answer.txt", @"the reference names the file the link resolves to");
	XCTAssertTrue([aliased.path hasPrefix:outputs.path], @"%@ keeps the directory's own spelling", aliased);
	XCTAssertEqualObjects([NSString stringWithContentsOfURL:aliased encoding:NSUTF8StringEncoding error:NULL], @"inside");
}

- (void)testAnIOSurfaceBufferFallsBackToSharedMemory
{
	NSDictionary *attributes = @{ (id)kCVPixelBufferIOSurfacePropertiesKey: @{} };
	CVPixelBufferRef source = [self newPixelBufferWithWidth:512 height:512 format:kCVPixelFormatType_32BGRA attributes:attributes];
	XCTAssertTrue(CVPixelBufferGetIOSurface(source) != NULL, @"the source is surface-backed");
	NSDictionary *entry = nil;
	NFKPluginAttachmentWriter *writer = nil;
	id decoded = [self roundTrip:(__bridge id)source entry:&entry writer:&writer];
	if (decoded == nil) {
		CVPixelBufferRelease(source);
		return;
	}
	XCTAssertEqualObjects(entry[@"$nfk"], @"buffer");
	XCTAssertEqualObjects(entry[@"as"], @"pixelBuffer", @"no surface crosses until the mach-surfaces capability");
	NSError *error = nil;
	NFKPluginSharedMemory *region = [NFKPluginSharedMemory regionWithDescriptor:dup(writer.descriptors.firstObject.intValue) copiesFiles:NO error:&error];
	XCTAssertTrue(region.isSharedMemoryObject, @"%@", error);
	XCTAssertTrue(CVPixelBufferGetIOSurface((__bridge CVPixelBufferRef)decoded) == NULL, @"the receiver wraps the shared bytes");
	[self assertPixelsOf:(__bridge CVPixelBufferRef)decoded equal:source];
	CVPixelBufferRelease(source);
}

- (void)testARegionMapsSharedMemoryCopiesFilesAndRefusesPipes
{
	NSError *error = nil;
	NFKPluginSharedMemory *made = [NFKPluginSharedMemory regionWithLength:8192 namePrefix:nil error:&error];
	XCTAssertTrue(made.isSharedMemoryObject, @"%@", error);
	memset(made.mutableBytes, 7, made.length);
	NFKPluginSharedMemory *mapped = [NFKPluginSharedMemory regionWithDescriptor:dup(made.descriptor) length:8192 copiesFiles:YES error:&error];
	XCTAssertFalse(mapped.isCopied, @"shared memory maps as it is");
	XCTAssertEqual(mapped.length, 8192u);
	if (mapped != nil) {
		XCTAssertEqual(((const uint8_t *)mapped.mutableBytes)[8191], 7);
	}

	NSURL *fileURL = [self.directory URLByAppendingPathComponent:@"region.bin"];
	XCTAssertTrue([[self randomDataOfLength:1000] writeToURL:fileURL atomically:YES]);
	int file = open(fileURL.fileSystemRepresentation, O_RDWR);
	NFKPluginSharedMemory *copied = [NFKPluginSharedMemory regionWithDescriptor:dup(file) copiesFiles:YES error:&error];
	XCTAssertTrue(copied.isCopied, @"a file from the peer is copied, so truncating it cannot fault a mapping");
	XCTAssertEqual(copied.descriptor, -1);
	NFKPluginSharedMemory *fileMapped = [NFKPluginSharedMemory regionWithDescriptor:file copiesFiles:NO error:&error];
	XCTAssertFalse(fileMapped.isCopied);
	XCTAssertEqualObjects(fileMapped.data, copied.data);

	int pipeDescriptors[2];
	XCTAssertEqual(pipe(pipeDescriptors), 0);
	XCTAssertNil([NFKPluginSharedMemory regionWithDescriptor:pipeDescriptors[0] copiesFiles:YES error:&error]);
	XCTAssertEqual(error.code, NFKPluginErrorProtocolViolation);
	close(pipeDescriptors[1]);
}

- (void)testARequestRidesAFrameWithItsAttachments
{
	int one = -1;
	int two = -1;
	NSError *error = nil;
	XCTAssertTrue([NFKPluginChannel createSocketPair:&one second:&two error:&error], @"%@", error);
	NFKPluginChannel *host = [[NFKPluginChannel alloc] initWithDescriptor:one];
	NFKPluginChannel *plugin = [[NFKPluginChannel alloc] initWithDescriptor:two];
	__block NSDictionary *received = nil;
	__block NSArray<NFKPluginDescriptor *> *receivedDescriptors = nil;
	XCTestExpectation *arrived = [self expectationWithDescription:@"the frame arrives"];
	plugin.messageHandler = ^(NSDictionary *message, NSArray<NFKPluginDescriptor *> *descriptors) {
		received = message;
		receivedDescriptors = descriptors;
		[arrived fulfill];
	};
	[plugin resume];

	NSData *image = [self randomDataOfLength:300 * 1024];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ @"blob": image, @"prompt": @"hello" }
															  parameters:@{ @"steps": @4 } outputModality:NFKModalityText];
	NFKPluginAttachmentWriter *writer NS_VALID_UNTIL_END_OF_SCOPE = [[NFKPluginAttachmentWriter alloc] init];
	id encoded = [[self encoderWithWriter:writer] JSONObjectForRequest:request error:&error];
	XCTAssertNotNil(encoded, @"%@", error);
	NSDictionary *run = @{ @"type": @"run", @"id": @1, @"request": encoded ?: @{} };
	XCTAssertTrue([host sendMessage:run descriptors:writer.descriptors error:&error], @"%@", error);
	[self waitForExpectations:@[ arrived ] timeout:5];

	NFKInferenceWireCoding *decoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:self.directory];
	NFKPluginAttachmentReader *reader = [[NFKPluginAttachmentReader alloc] initWithDescriptors:receivedDescriptors];
	reader.copiesFileRegions = NO;
	decoder.attachmentSource = reader;
	NFKInferenceRequest *decoded = [decoder requestFromJSONObject:received[@"request"] error:&error];
	XCTAssertNotNil(decoded, @"%@", error);
	XCTAssertEqualObjects(decoded.inputs[@"blob"], image);
	XCTAssertEqualObjects(decoded.inputs[@"prompt"], @"hello");
	XCTAssertEqualObjects(decoded.parameters[@"steps"], @4);
	XCTAssertEqual(decoded.outputModality, NFKModalityText);
	[host close];
	[plugin close];
}

#pragma mark Lengths that are not whole pages

- (void)testRegionsOfLengthsOffPageBoundariesRoundTrip
{
	for (NSNumber *size in @[ @1, @4097, @16383, @16385 ]) {
		NSUInteger length = size.unsignedIntegerValue;
		NSError *error = nil;
		NFKPluginSharedMemory *made = [NFKPluginSharedMemory regionWithLength:length namePrefix:nil error:&error];
		XCTAssertNotNil(made, @"%@", error);
		if (made == nil) {
			continue;
		}
		XCTAssertEqual(made.length, length, @"the creator keeps the length it asked for");
		XCTAssertGreaterThanOrEqual(made.mappedLength, length);
		XCTAssertEqual(made.mappedLength % vm_page_size, 0u, @"the object is whole pages");
		for (NSUInteger index = 0; index < length; index++) {
			((uint8_t *)made.mutableBytes)[index] = (uint8_t)(index % 251 + 1);
		}
		XCTAssertEqual(made.data.length, length);

		NFKPluginSharedMemory *whole = [NFKPluginSharedMemory regionWithDescriptor:dup(made.descriptor) copiesFiles:YES error:&error];
		XCTAssertGreaterThanOrEqual(whole.length, length, @"without a stated length a receiver sees the whole object, %@", error);
		NFKPluginSharedMemory *stated = [NFKPluginSharedMemory regionWithDescriptor:dup(made.descriptor) length:length copiesFiles:YES error:&error];
		XCTAssertEqual(stated.length, length, @"%@", error);
		XCTAssertEqualObjects(stated.data, made.data, @"%lu bytes", (unsigned long)length);
		XCTAssertNil([NFKPluginSharedMemory regionWithDescriptor:dup(made.descriptor) length:whole.length + 1 copiesFiles:YES error:&error],
					 @"a length beyond the object is refused");
		XCTAssertEqual(error.code, NFKPluginErrorProtocolViolation);

		NFKPluginAttachmentWriter *writer NS_VALID_UNTIL_END_OF_SCOPE = [[NFKPluginAttachmentWriter alloc] init];
		NFKInferenceWireCoding *encoder = [self encoderWithWriter:writer];
		encoder.inlineByteLimit = 0;
		NSData *data = [self randomDataOfLength:length];
		NSDictionary *entry = [encoder JSONValueForValue:data key:@"value" error:&error];
		XCTAssertEqualObjects(entry[@"length"], @(length));
		NFKInferenceWireCoding *decoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:self.directory];
		decoder.attachmentSource = [self readerForWriter:writer];
		NSData *decoded = [decoder valueFromJSONValue:entry key:@"value" error:&error];
		XCTAssertEqual(decoded.length, length, @"the decoder reads the entry's length of a whole-page region, %@", error);
		XCTAssertEqualObjects(decoded, data);
	}
}

- (void)testAMalformedAttachmentFailsCleanly
{
	NFKPluginAttachmentWriter *writer NS_VALID_UNTIL_END_OF_SCOPE = [[NFKPluginAttachmentWriter alloc] init];
	NSInteger index = -1;
	NSError *error = nil;
	XCTAssertTrue([writer attachmentRegionOfLength:4097 index:&index error:&error] != NULL, @"%@", error);
	NFKInferenceWireCoding *decoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:self.directory];
	decoder.attachmentSource = [self readerForWriter:writer];
	NSDictionary *plane = @{ @"offset": @0, @"bytesPerRow": @256, @"rows": @64 };
	NSDictionary *pixels = @{ @"$nfk": @"buffer", @"index": @0, @"as": @"pixelBuffer", @"length": @4097,
							  @"format": @(kCVPixelFormatType_32BGRA), @"width": @64, @"height": @64 };
	NSMutableDictionary *pastTheEnd = [pixels mutableCopy];
	pastTheEnd[@"planes"] = @[ plane ];
	NSMutableDictionary *fractional = [pixels mutableCopy];
	fractional[@"height"] = @1;
	fractional[@"planes"] = @[ @{ @"offset": @0, @"bytesPerRow": @256.5, @"rows": @1 } ];
	NSMutableDictionary *offsetPastTheEnd = [pixels mutableCopy];
	offsetPastTheEnd[@"height"] = @1;
	offsetPastTheEnd[@"planes"] = @[ @{ @"offset": @4096, @"bytesPerRow": @256, @"rows": @1 } ];
	NSDictionary *cases = @{
		@"a plane past the attachment's end": pastTheEnd,
		@"a fractional stride": fractional,
		@"a plane that starts too late": offsetPastTheEnd,
		@"a length past the region": @{ @"$nfk": @"buffer", @"index": @0, @"as": @"data", @"length": @1000000 },
		@"no length": @{ @"$nfk": @"buffer", @"index": @0, @"as": @"data" },
		@"a multi-array longer than its attachment": @{ @"$nfk": @"buffer", @"index": @0, @"as": @"multiArray", @"length": @4096,
														@"dataType": @(MLMultiArrayDataTypeFloat32), @"shape": @[ @2048 ] },
	};
	[cases enumerateKeysAndObjectsUsingBlock:^(NSString *name, NSDictionary *entry, BOOL *stop) {
		NSError *failure = nil;
		XCTAssertNil([decoder valueFromJSONValue:entry key:@"value" error:&failure], @"%@ is refused", name);
		XCTAssertEqual(failure.code, kNFKError_InferenceUnsupported, @"%@ fails with an error", name);
	}];
	NSMutableDictionary *fits = [pixels mutableCopy];
	fits[@"height"] = @16;
	fits[@"planes"] = @[ @{ @"offset": @0, @"bytesPerRow": @256, @"rows": @16 } ];
	XCTAssertNotNil([decoder valueFromJSONValue:fits key:@"value" error:&error], @"a plane inside the length wraps, %@", error);
}

- (void)testAMalformedPCMAttachmentFailsCleanly
{
	AVAudioFormat *planar = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32 sampleRate:48000 channels:2 interleaved:NO];
	AVAudioPCMBuffer *source = [[AVAudioPCMBuffer alloc] initWithPCMFormat:planar frameCapacity:20000];
	source.frameLength = 20000;
	NFKPluginAttachmentWriter *writer = nil;
	NSDictionary *entry = nil;
	AVAudioPCMBuffer *decoded = [self roundTrip:source entry:&entry writer:&writer];
	if (![decoded isKindOfClass:AVAudioPCMBuffer.class] || [entry[@"buffers"] count] != 2) {
		XCTFail(@"the well-formed buffer decodes, %@", entry);
		return;
	}
	NFKInferenceWireCoding *decoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:self.directory];
	decoder.attachmentSource = [self readerForWriter:writer];
	NSDictionary *buffer = [entry[@"buffers"] firstObject];
	NSDictionary<NSString *, NSDictionary *> *changes = @{
		@"a fractional frame count": @{ @"frameLength": @19999.5 },
		@"no frame count": @{ @"frameLength": NSNull.null },
		@"one buffer for two channels": @{ @"buffers": @[ buffer ] },
		@"a buffer of the wrong length": @{ @"buffers": @[ buffer, @{ @"offset": buffer[@"offset"], @"length": @1 } ] },
		@"a misaligned buffer": @{ @"buffers": @[ buffer, @{ @"offset": @1, @"length": buffer[@"length"] } ] },
		@"a buffer past the attachment": @{ @"buffers": @[ buffer, @{ @"offset": entry[@"length"], @"length": buffer[@"length"] } ] },
		@"a frame size the format does not have": @{ @"bytesPerFrame": @2 },
		@"more frames than the buffers hold": @{ @"frameLength": @40000 },
	};
	[changes enumerateKeysAndObjectsUsingBlock:^(NSString *name, NSDictionary *change, BOOL *stop) {
		NSMutableDictionary *malformed = [entry mutableCopy];
		[change enumerateKeysAndObjectsUsingBlock:^(NSString *field, id value, BOOL *innerStop) {
			malformed[field] = value == NSNull.null ? nil : value;
		}];
		NSError *failure = nil;
		XCTAssertNil([decoder valueFromJSONValue:malformed key:@"value" error:&failure], @"%@ is refused", name);
		XCTAssertEqual(failure.code, kNFKError_InferenceUnsupported, @"%@ fails with an error", name);
	}];
	@autoreleasepool {
		for (NSUInteger round = 0; round < 50; round++) {
			NSError *error = nil;
			AVAudioPCMBuffer *again = [decoder valueFromJSONValue:entry key:@"value" error:&error];
			XCTAssertEqual(again.frameLength, 20000u, @"%@", error);
		}
	}
	NSMutableData *afterwards = [NSMutableData dataWithLength:1 << 20];
	XCTAssertEqual(afterwards.length, 1u << 20, @"releasing wrapped buffers leaves the heap intact");
}

#pragma mark The HTTP form

- (void)testTheHTTPFormIsUnchangedWithoutASink
{
	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:self.directory];
	NSError *error = nil;

	id small = [coder JSONValueForValue:[@"abc" dataUsingEncoding:NSUTF8StringEncoding] key:@"value" error:&error];
	NSData *text = [NSJSONSerialization dataWithJSONObject:small options:NSJSONWritingSortedKeys error:NULL];
	XCTAssertEqualObjects([[NSString alloc] initWithData:text encoding:NSUTF8StringEncoding], @"{\"$nfk\":\"data\",\"base64\":\"YWJj\"}");

	NSData *large = [self randomDataOfLength:200 * 1024];
	NSDictionary *expectedData = @{ @"$nfk": @"data", @"base64": [large base64EncodedStringWithOptions:0] };
	XCTAssertEqualObjects([coder JSONValueForValue:large key:@"value" error:&error], expectedData, @"no size moves a value out of the JSON");

	NSMutableArray<NSNumber *> *floats = [NSMutableArray array];
	NSMutableData *packed = [NSMutableData dataWithLength:30000 * sizeof(float)];
	for (NSUInteger index = 0; index < 30000; index++) {
		[floats addObject:@((float)index)];
		((float *)packed.mutableBytes)[index] = (float)index;
	}
	NSDictionary *expectedFloats = @{ @"$nfk": @"float32Array", @"base64": [packed base64EncodedStringWithOptions:0] };
	XCTAssertEqualObjects([coder JSONValueForValue:floats key:@"value" error:&error], expectedFloats);

	CVPixelBufferRef buffer = [self newPixelBufferWithWidth:250 height:96 format:kCVPixelFormatType_32BGRA
												 attributes:@{ (id)kCVPixelBufferBytesPerRowAlignmentKey: @256 }];
	CVPixelBufferLockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
	size_t stride = CVPixelBufferGetBytesPerRow(buffer);
	NSData *rows = [NSData dataWithBytes:CVPixelBufferGetBaseAddress(buffer) length:stride * 96];
	CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
	NSDictionary *expectedPixels = @{ @"$nfk": @"pixelBuffer", @"format": @(kCVPixelFormatType_32BGRA), @"width": @250, @"height": @96,
									  @"planes": @[ @{ @"bytesPerRow": @(stride), @"rows": @96, @"base64": [rows base64EncodedStringWithOptions:0] } ] };
	XCTAssertEqualObjects([coder JSONValueForValue:(__bridge id)buffer key:@"value" error:&error], expectedPixels);
	CVPixelBufferRelease(buffer);

	NSURL *file = [self.directory URLByAppendingPathComponent:@"notes.txt"];
	XCTAssertTrue([[@"plain" dataUsingEncoding:NSUTF8StringEncoding] writeToURL:file atomically:YES]);
	NSDictionary *expectedFile = @{ @"$nfk": @"file", @"name": @"notes.txt", @"base64": @"cGxhaW4=" };
	XCTAssertEqualObjects([coder JSONValueForValue:file key:@"value" error:&error], expectedFile, @"a file still crosses by its bytes");

	NSDictionary *attachment = @{ @"$nfk": @"buffer", @"index": @0, @"as": @"data" };
	XCTAssertNil([coder valueFromJSONValue:attachment key:@"value" error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported, @"an attachment means nothing to the HTTP route");
	NSDictionary *reference = @{ @"$nfk": @"file", @"path": file.path, @"name": @"notes.txt" };
	XCTAssertNil([coder valueFromJSONValue:reference key:@"value" error:&error],
				 @"a client cannot name a path on the server");
	XCTAssertTrue([error.localizedDescription containsString:@"carries no bytes"], @"%@", error);
}

@end

#endif
