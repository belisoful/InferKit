//
//  NFKValueFiles.m
//  InferKit
//

#import "NFKValueFiles.h"
#import <InferKit/NFKErrors.h>
#import <ImageIO/ImageIO.h>
#import <CoreGraphics/CoreGraphics.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <limits.h>

static const NSUInteger NFKValueFileStemLimit = 40;

static BOOL NFKValueFileFail(NSError **error, NSString *reason)
{
	if (error != NULL) {
		*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceUnsupported
								 userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return NO;
}

#pragma mark Names and paths

static NSString *NFKValueFileSafeStem(NSString *stem)
{
	NSMutableString *safe = [NSMutableString string];
	NSCharacterSet *allowed = [NSCharacterSet characterSetWithCharactersInString:
							   @"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"];
	for (NSUInteger index = 0; index < stem.length && safe.length < NFKValueFileStemLimit; index++) {
		unichar character = [stem characterAtIndex:index];
		[safe appendString:[allowed characterIsMember:character] ? [NSString stringWithCharacters:&character length:1] : @"_"];
	}
	NSString *trimmed = [safe stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@"_-"]];
	return trimmed.length > 0 ? trimmed : @"value";
}

NSURL *NFKValueFileCreate(NSURL *directory, NSString *stem, NSString *extension, NSError **error)
{
	NSString *safe = NFKValueFileSafeStem(stem);
	for (NSUInteger attempt = 1; attempt < 100000; attempt++) {
		NSString *name = attempt == 1 ? safe : [NSString stringWithFormat:@"%@-%lu", safe, (unsigned long)attempt];
		NSURL *url = [directory URLByAppendingPathComponent:[name stringByAppendingPathExtension:extension] isDirectory:NO];
		int descriptor = open(url.fileSystemRepresentation, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0644);
		if (descriptor >= 0) {
			close(descriptor);
			return url;
		}
		if (errno != EEXIST) {
			NFKValueFileFail(error, [NSString stringWithFormat:@"no file can be made in %@: %s", directory.path, strerror(errno)]);
			return nil;
		}
	}
	NFKValueFileFail(error, [NSString stringWithFormat:@"every name for %@ is taken in %@", safe, directory.path]);
	return nil;
}

static NSString * _Nullable NFKValueFileRealPath(NSURL *url)
{
	char resolved[PATH_MAX];
	return realpath(url.fileSystemRepresentation, resolved) != NULL ? [NSString stringWithUTF8String:resolved] : nil;
}

NSString *NFKValueFileRelativePath(NSURL *directory, NSURL *url)
{
	NSString *root = NFKValueFileRealPath(directory);
	NSString *path = NFKValueFileRealPath(url);
	if (root == nil || path == nil || ![path hasPrefix:[root stringByAppendingString:@"/"]]) {
		return nil;
	}
	return [path substringFromIndex:root.length + 1];
}

NSURL *NFKValueFileResolve(NSURL *directory, NSString *path, NSError **error)
{
	if (path.length == 0 || path.isAbsolutePath) {
		NFKValueFileFail(error, [NSString stringWithFormat:@"the path \"%@\" is not relative to the value directory", path]);
		return nil;
	}
	NSURL *url = [directory URLByAppendingPathComponent:path isDirectory:NO];
	if (NFKValueFileRelativePath(directory, url) == nil) {
		NFKValueFileFail(error, [NSString stringWithFormat:@"the path \"%@\" names no file inside the value directory", path]);
		return nil;
	}
	return url;
}

#pragma mark WAV

static const uint16_t NFKWAVFormatPCM = 1;
static const uint16_t NFKWAVFormatFloat = 3;
static const uint16_t NFKWAVFormatExtensible = 0xFFFE;

BOOL NFKValueFileWAVHoldsFormat(AVAudioFormat *format)
{
	const AudioStreamBasicDescription *description = format.streamDescription;
	if (description == NULL || description->mFormatID != kAudioFormatLinearPCM
		|| (description->mFormatFlags & kAudioFormatFlagIsBigEndian) != 0) {
		return NO;
	}
	switch (format.commonFormat) {
		case AVAudioPCMFormatFloat32:
		case AVAudioPCMFormatFloat64:
		case AVAudioPCMFormatInt16:
		case AVAudioPCMFormatInt32:
			return YES;
		default:
			return NO;
	}
}

static void NFKWAVAppend16(NSMutableData *data, uint16_t value)
{
	uint8_t bytes[2] = { (uint8_t)value, (uint8_t)(value >> 8) };
	[data appendBytes:bytes length:2];
}

static void NFKWAVAppend32(NSMutableData *data, uint32_t value)
{
	uint8_t bytes[4] = { (uint8_t)value, (uint8_t)(value >> 8), (uint8_t)(value >> 16), (uint8_t)(value >> 24) };
	[data appendBytes:bytes length:4];
}

static uint16_t NFKWAVRead16(const uint8_t *bytes)
{
	return (uint16_t)(bytes[0] | (bytes[1] << 8));
}

static uint32_t NFKWAVRead32(const uint8_t *bytes)
{
	return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) | ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static NSUInteger NFKWAVSampleSize(AVAudioFormat *format)
{
	switch (format.commonFormat) {
		case AVAudioPCMFormatInt16:
			return 2;
		case AVAudioPCMFormatFloat64:
			return 8;
		default:
			return 4;
	}
}

NSData *NFKValueFileWAVData(AVAudioPCMBuffer *buffer, NSError **error)
{
	AVAudioFormat *format = buffer.format;
	if (!NFKValueFileWAVHoldsFormat(format)) {
		NFKValueFileFail(error, @"WAV does not hold this PCM format");
		return nil;
	}
	NSUInteger sampleSize = NFKWAVSampleSize(format);
	NSUInteger channels = format.channelCount;
	NSUInteger frames = buffer.frameLength;
	unsigned long long payload = (unsigned long long)frames * channels * sampleSize;
	if (payload > UINT32_MAX - 64) {
		NFKValueFileFail(error, @"the PCM buffer is larger than a WAV file holds");
		return nil;
	}
	BOOL floating = format.commonFormat == AVAudioPCMFormatFloat32 || format.commonFormat == AVAudioPCMFormatFloat64;
	NSMutableData *data = [NSMutableData dataWithCapacity:(NSUInteger)payload + 44];
	[data appendBytes:"RIFF" length:4];
	NFKWAVAppend32(data, (uint32_t)(payload + 36));
	[data appendBytes:"WAVEfmt " length:8];
	NFKWAVAppend32(data, 16);
	NFKWAVAppend16(data, floating ? NFKWAVFormatFloat : NFKWAVFormatPCM);
	NFKWAVAppend16(data, (uint16_t)channels);
	NFKWAVAppend32(data, (uint32_t)llround(format.sampleRate));
	NFKWAVAppend32(data, (uint32_t)(llround(format.sampleRate) * channels * sampleSize));
	NFKWAVAppend16(data, (uint16_t)(channels * sampleSize));
	NFKWAVAppend16(data, (uint16_t)(sampleSize * 8));
	[data appendBytes:"data" length:4];
	NFKWAVAppend32(data, (uint32_t)payload);
	const AudioBufferList *list = buffer.audioBufferList;
	if (format.isInterleaved || channels == 1) {
		[data appendBytes:list->mBuffers[0].mData length:(NSUInteger)payload];
		return data;
	}
	NSUInteger start = data.length;
	[data increaseLengthBy:(NSUInteger)payload];
	uint8_t *destination = (uint8_t *)data.mutableBytes + start;
	for (NSUInteger channel = 0; channel < channels; channel++) {
		const uint8_t *source = list->mBuffers[channel].mData;
		for (NSUInteger frame = 0; frame < frames; frame++) {
			memcpy(destination + (frame * channels + channel) * sampleSize, source + frame * sampleSize, sampleSize);
		}
	}
	return data;
}

AVAudioPCMBuffer *NFKValueFilePCMBufferFromWAV(NSData *data, AVAudioFormat *format, NSError **error)
{
	const uint8_t *bytes = data.bytes;
	NSUInteger length = data.length;
	if (length < 12 || memcmp(bytes, "RIFF", 4) != 0 || memcmp(bytes + 8, "WAVE", 4) != 0) {
		NFKValueFileFail(error, @"the file is not a WAV file");
		return nil;
	}
	uint16_t tag = 0;
	uint16_t channels = 0;
	uint32_t rate = 0;
	uint16_t bits = 0;
	const uint8_t *samples = NULL;
	NSUInteger sampleBytes = 0;
	NSUInteger offset = 12;
	while (offset + 8 <= length) {
		uint32_t chunkLength = NFKWAVRead32(bytes + offset + 4);
		if (chunkLength > length - offset - 8) {
			NFKValueFileFail(error, @"a WAV chunk runs past the file");
			return nil;
		}
		const uint8_t *chunk = bytes + offset + 8;
		if (memcmp(bytes + offset, "fmt ", 4) == 0 && chunkLength >= 16) {
			tag = NFKWAVRead16(chunk);
			channels = NFKWAVRead16(chunk + 2);
			rate = NFKWAVRead32(chunk + 4);
			bits = NFKWAVRead16(chunk + 14);
			if (tag == NFKWAVFormatExtensible && chunkLength >= 26) {
				tag = NFKWAVRead16(chunk + 24);
			}
		} else if (memcmp(bytes + offset, "data", 4) == 0) {
			samples = chunk;
			sampleBytes = chunkLength;
		}
		offset += 8 + chunkLength + (chunkLength & 1);
	}
	NSUInteger sampleSize = NFKWAVSampleSize(format);
	BOOL floating = format.commonFormat == AVAudioPCMFormatFloat32 || format.commonFormat == AVAudioPCMFormatFloat64;
	BOOL matches = NFKValueFileWAVHoldsFormat(format) && samples != NULL && channels == format.channelCount
		&& rate == (uint32_t)llround(format.sampleRate) && bits == sampleSize * 8
		&& tag == (floating ? NFKWAVFormatFloat : NFKWAVFormatPCM);
	if (!matches) {
		NFKValueFileFail(error, @"the WAV file's samples do not match the PCM format its entry names");
		return nil;
	}
	NSUInteger frameSize = channels * sampleSize;
	AVAudioFrameCount frames = (AVAudioFrameCount)(sampleBytes / frameSize);
	AVAudioPCMBuffer *buffer = [[AVAudioPCMBuffer alloc] initWithPCMFormat:format frameCapacity:MAX(frames, 1)];
	if (buffer == nil) {
		NFKValueFileFail(error, @"no PCM buffer of the WAV file's length can be made");
		return nil;
	}
	AudioBufferList *list = buffer.mutableAudioBufferList;
	if (format.isInterleaved || channels == 1) {
		memcpy(list->mBuffers[0].mData, samples, (size_t)frames * frameSize);
	} else {
		for (NSUInteger channel = 0; channel < channels; channel++) {
			uint8_t *destination = list->mBuffers[channel].mData;
			for (NSUInteger frame = 0; frame < frames; frame++) {
				memcpy(destination + frame * sampleSize, samples + frame * frameSize + channel * sampleSize, sampleSize);
			}
		}
	}
	buffer.frameLength = frames;
	return buffer;
}

#pragma mark NumPy

NSString *NFKValueFileNPYDescription(BOOL floating, NSUInteger elementSize)
{
	if (floating) {
		return elementSize == 2 || elementSize == 4 || elementSize == 8 ? [NSString stringWithFormat:@"<f%lu", (unsigned long)elementSize] : nil;
	}
	if (elementSize == 1) {
		return @"|i1";
	}
	return elementSize == 2 || elementSize == 4 || elementSize == 8 ? [NSString stringWithFormat:@"<i%lu", (unsigned long)elementSize] : nil;
}

NSData *NFKValueFileNPYData(NSData *elements, NSString *description, NSArray<NSNumber *> *shape)
{
	NSMutableArray<NSString *> *extents = [NSMutableArray arrayWithCapacity:shape.count];
	for (NSNumber *extent in shape) {
		[extents addObject:extent.stringValue];
	}
	NSString *tuple = shape.count == 1 ? [NSString stringWithFormat:@"(%@,)", extents.firstObject]
		: [NSString stringWithFormat:@"(%@)", [extents componentsJoinedByString:@", "]];
	NSMutableString *header = [NSMutableString stringWithFormat:@"{'descr': '%@', 'fortran_order': False, 'shape': %@, }", description, tuple];
	// The magic, the version, and the header length take 10 bytes, and the whole preamble ends on a
	// 64-byte boundary with a newline.
	while ((10 + header.length + 1) % 64 != 0) {
		[header appendString:@" "];
	}
	[header appendString:@"\n"];
	NSMutableData *data = [NSMutableData dataWithCapacity:10 + header.length + elements.length];
	const uint8_t magic[8] = { 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0 };
	[data appendBytes:magic length:8];
	NFKWAVAppend16(data, (uint16_t)header.length);
	[data appendData:[header dataUsingEncoding:NSASCIIStringEncoding]];
	[data appendData:elements];
	return data;
}

/*! The text of a header field's value: the quoted string after 'name':, or the parenthesized tuple. */
static NSString * _Nullable NFKNPYField(NSString *header, NSString *name)
{
	NSRange key = [header rangeOfString:[NSString stringWithFormat:@"'%@':", name]];
	if (key.location == NSNotFound) {
		return nil;
	}
	NSString *rest = [[header substringFromIndex:NSMaxRange(key)] stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
	NSString *closing = [rest hasPrefix:@"'"] ? @"'" : ([rest hasPrefix:@"("] ? @")" : @",");
	NSUInteger start = [rest hasPrefix:@"'"] || [rest hasPrefix:@"("] ? 1 : 0;
	NSRange end = [rest rangeOfString:closing options:0 range:NSMakeRange(start, rest.length - start)];
	return end.location == NSNotFound ? nil : [rest substringWithRange:NSMakeRange(start, end.location - start)];
}

BOOL NFKValueFileReadNPY(NSData *data, NSData **outElements, NSString **outDescription, NSArray<NSNumber *> **outShape, NSError **error)
{
	const uint8_t *bytes = data.bytes;
	if (data.length < 10 || memcmp(bytes, "\x93NUMPY", 6) != 0) {
		return NFKValueFileFail(error, @"the file is not a NumPy .npy file");
	}
	NSUInteger headerStart = bytes[6] == 1 ? 10 : 12;
	if (data.length < headerStart) {
		return NFKValueFileFail(error, @"the .npy header is cut short");
	}
	NSUInteger headerLength = bytes[6] == 1 ? NFKWAVRead16(bytes + 8) : NFKWAVRead32(bytes + 8);
	if (headerLength > data.length - headerStart) {
		return NFKValueFileFail(error, @"the .npy header runs past the file");
	}
	NSString *header = [[NSString alloc] initWithBytes:bytes + headerStart length:headerLength encoding:NSASCIIStringEncoding];
	NSString *description = header != nil ? NFKNPYField(header, @"descr") : nil;
	NSString *order = header != nil ? NFKNPYField(header, @"fortran_order") : nil;
	NSString *tuple = header != nil ? NFKNPYField(header, @"shape") : nil;
	if (description == nil || tuple == nil || ![order hasPrefix:@"False"]) {
		return NFKValueFileFail(error, @"the .npy header names no type, no shape, or a Fortran order");
	}
	NSMutableArray<NSNumber *> *shape = [NSMutableArray array];
	unsigned long long count = 1;
	for (NSString *part in [tuple componentsSeparatedByString:@","]) {
		NSString *trimmed = [part stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
		if (trimmed.length == 0) {
			continue;
		}
		long long extent = trimmed.longLongValue;
		if (extent < 0 || ![[NSString stringWithFormat:@"%lld", extent] isEqualToString:trimmed]) {
			return NFKValueFileFail(error, @"the .npy shape is malformed");
		}
		[shape addObject:@(extent)];
		if (count != ULLONG_MAX && __builtin_mul_overflow(count, (unsigned long long)extent, &count)) {
			count = ULLONG_MAX;
		}
	}
	NSUInteger elementSize = (NSUInteger)[description substringFromIndex:2].integerValue;
	BOOL known = description.length >= 3 && ([description hasPrefix:@"<"] || [description hasPrefix:@"|"])
		&& [NFKValueFileNPYDescription([description characterAtIndex:1] == 'f', elementSize) isEqualToString:description];
	NSUInteger payloadStart = headerStart + headerLength;
	unsigned long long payload = 0;
	if (!known || count == ULLONG_MAX || __builtin_mul_overflow(count, (unsigned long long)elementSize, &payload)
		|| payload != data.length - payloadStart) {
		return NFKValueFileFail(error, [NSString stringWithFormat:@"the .npy file's type %@ and shape do not match its bytes", description]);
	}
	*outElements = [data subdataWithRange:NSMakeRange(payloadStart, data.length - payloadStart)];
	*outDescription = description;
	*outShape = shape;
	return YES;
}

#pragma mark PNG

BOOL NFKValueFilePNGHoldsPixelFormat(OSType format)
{
	return format == kCVPixelFormatType_32BGRA || format == kCVPixelFormatType_32ARGB
		|| format == kCVPixelFormatType_OneComponent8;
}

/*! The positions of red, green, blue, and alpha within a four-byte pixel of format. */
static void NFKValueFileChannelOrder(OSType format, int order[4])
{
	int bgra[4] = { 2, 1, 0, 3 };
	int rgba[4] = { 0, 1, 2, 3 };
	int argb[4] = { 1, 2, 3, 0 };
	memcpy(order, format == kCVPixelFormatType_32BGRA ? bgra : (format == kCVPixelFormatType_32ARGB ? argb : rgba), sizeof(int) * 4);
}

NSData *NFKValueFilePNGData(CVPixelBufferRef buffer)
{
	OSType format = CVPixelBufferGetPixelFormatType(buffer);
	if (!NFKValueFilePNGHoldsPixelFormat(format) || CVPixelBufferLockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess) {
		return nil;
	}
	size_t width = CVPixelBufferGetWidth(buffer);
	size_t height = CVPixelBufferGetHeight(buffer);
	size_t bytesPerRow = CVPixelBufferGetBytesPerRow(buffer);
	const uint8_t *base = CVPixelBufferGetBaseAddress(buffer);
	BOOL gray = format == kCVPixelFormatType_OneComponent8;
	size_t components = gray ? 1 : 4;
	NSMutableData *pixels = [NSMutableData dataWithLength:width * height * components];
	uint8_t *destination = pixels.mutableBytes;
	int order[4];
	NFKValueFileChannelOrder(format, order);
	for (size_t row = 0; row < height; row++) {
		const uint8_t *source = base + row * bytesPerRow;
		uint8_t *line = destination + row * width * components;
		if (gray) {
			memcpy(line, source, width);
			continue;
		}
		for (size_t column = 0; column < width; column++) {
			for (int channel = 0; channel < 4; channel++) {
				line[column * 4 + (size_t)channel] = source[column * 4 + (size_t)order[channel]];
			}
		}
	}
	CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
	CGColorSpaceRef space = gray ? CGColorSpaceCreateWithName(kCGColorSpaceGenericGrayGamma2_2) : CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
	CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)pixels);
	CGBitmapInfo info = gray ? (CGBitmapInfo)kCGImageAlphaNone : (CGBitmapInfo)kCGImageAlphaLast;
	CGImageRef image = CGImageCreate(width, height, 8, 8 * components, width * components, space, info, provider, NULL, false,
									 kCGRenderingIntentDefault);
	CGDataProviderRelease(provider);
	CGColorSpaceRelease(space);
	if (image == NULL) {
		return nil;
	}
	NSMutableData *png = [NSMutableData data];
	CGImageDestinationRef destinationFile = CGImageDestinationCreateWithData((__bridge CFMutableDataRef)png, CFSTR("public.png"), 1, NULL);
	BOOL written = NO;
	if (destinationFile != NULL) {
		CGImageDestinationAddImage(destinationFile, image, NULL);
		written = CGImageDestinationFinalize(destinationFile);
		CFRelease(destinationFile);
	}
	CGImageRelease(image);
	return written ? png : nil;
}

CVPixelBufferRef NFKValueFilePixelBufferFromPNG(NSData *data, OSType format, NSError **error)
{
	CGImageSourceRef source = CGImageSourceCreateWithData((__bridge CFDataRef)data, NULL);
	CGImageRef image = source != NULL ? CGImageSourceCreateImageAtIndex(source, 0, NULL) : NULL;
	if (source != NULL) {
		CFRelease(source);
	}
	if (image == NULL) {
		NFKValueFileFail(error, @"the PNG could not be decoded");
		return NULL;
	}
	size_t width = CGImageGetWidth(image);
	size_t height = CGImageGetHeight(image);
	size_t bitsPerPixel = CGImageGetBitsPerPixel(image);
	CGImageAlphaInfo alpha = CGImageGetAlphaInfo(image);
	CGBitmapInfo byteOrder = CGImageGetBitmapInfo(image) & kCGBitmapByteOrderMask;
	size_t components = bitsPerPixel / 8;
	BOOL straight = alpha == kCGImageAlphaNone || alpha == kCGImageAlphaLast || alpha == kCGImageAlphaNoneSkipLast;
	BOOL laidOut = CGImageGetBitsPerComponent(image) == 8 && straight
		&& (byteOrder == kCGBitmapByteOrderDefault || byteOrder == kCGBitmapByteOrder32Big)
		&& (components == 1 || components == 3 || components == 4);
	CFDataRef decoded = laidOut ? CGDataProviderCopyData(CGImageGetDataProvider(image)) : NULL;
	size_t sourceBytesPerRow = CGImageGetBytesPerRow(image);
	CGImageRelease(image);
	if (decoded == NULL || (size_t)CFDataGetLength(decoded) < sourceBytesPerRow * height) {
		if (decoded != NULL) {
			CFRelease(decoded);
		}
		NFKValueFileFail(error, @"the PNG holds samples of a layout this form does not read");
		return NULL;
	}
	NSDictionary *attributes = @{ (id)kCVPixelBufferIOSurfacePropertiesKey: @{}, (id)kCVPixelBufferMetalCompatibilityKey: @YES };
	CVPixelBufferRef buffer = NULL;
	if (CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, (__bridge CFDictionaryRef)attributes, &buffer) != kCVReturnSuccess) {
		CFRelease(decoded);
		NFKValueFileFail(error, [NSString stringWithFormat:@"no pixel buffer of format %u can be made here", (unsigned)format]);
		return NULL;
	}
	CVPixelBufferLockBaseAddress(buffer, 0);
	uint8_t *base = CVPixelBufferGetBaseAddress(buffer);
	size_t bytesPerRow = CVPixelBufferGetBytesPerRow(buffer);
	const uint8_t *pixels = CFDataGetBytePtr(decoded);
	int order[4];
	NFKValueFileChannelOrder(format, order);
	BOOL hasAlpha = alpha == kCGImageAlphaLast;
	for (size_t row = 0; row < height; row++) {
		const uint8_t *source = pixels + row * sourceBytesPerRow;
		uint8_t *destination = base + row * bytesPerRow;
		for (size_t column = 0; column < width; column++) {
			const uint8_t *pixel = source + column * components;
			uint8_t red = pixel[0];
			uint8_t green = components >= 3 ? pixel[1] : pixel[0];
			uint8_t blue = components >= 3 ? pixel[2] : pixel[0];
			uint8_t opacity = hasAlpha && components == 4 ? pixel[3] : 255;
			if (format == kCVPixelFormatType_OneComponent8) {
				destination[column] = components >= 3 ? (uint8_t)((red * 299 + green * 587 + blue * 114 + 500) / 1000) : red;
				continue;
			}
			uint8_t values[4] = { red, green, blue, opacity };
			for (int channel = 0; channel < 4; channel++) {
				destination[column * 4 + (size_t)order[channel]] = values[channel];
			}
		}
	}
	CVPixelBufferUnlockBaseAddress(buffer, 0);
	CFRelease(decoded);
	return buffer;
}
