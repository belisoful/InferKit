//
//  NFKInferenceWireCoding.m
//  InferKit
//

#import "NFKInferenceWireCoding.h"
#import "NFKRemoteMediaSupport.h"
#import <InferKit/NFKAudioAsset.h>
#import <InferKit/NFKVideoAsset.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKDecisionQuestion.h>
#import <InferKit/NFKDecisionAnswer.h>
#import <InferKit/NFKDetection.h>
#import <InferKit/NFKKeypoint.h>
#import <InferKit/NFKClassification.h>
#import <InferKit/NFKAudioSegment.h>
#import <InferKit/NFKMIDISequence.h>
#import <InferKit/NFKMIDINote.h>
#import <InferKit/NFKMusicBeat.h>
#import <InferKit/NFKQuadrilateral.h>
#import <InferKit/NFKErrors.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>

static NSString * const NFKWireTag = @"$nfk";

/*! A number stored as a float or a double, as opposed to an integer or a boolean. */
static BOOL NFKWireIsFloating(NSNumber *number)
{
	const char *type = number.objCType;
	return CFGetTypeID((__bridge CFTypeRef)number) != CFBooleanGetTypeID()
		&& (strcmp(type, @encode(double)) == 0 || strcmp(type, @encode(float)) == 0);
}

static BOOL NFKWireIsFloatingArray(NSArray *array)
{
	if (array.count == 0) {
		return NO;
	}
	for (id element in array) {
		if (![element isKindOfClass:NSNumber.class] || !NFKWireIsFloating(element) || !isfinite([element doubleValue])) {
			return NO;
		}
	}
	return YES;
}

NSString *NFKServedFileExtension(NSString * _Nullable extension)
{
	NSString *lowered = extension.lowercaseString ?: @"";
	NSCharacterSet *invalid = [NSCharacterSet characterSetWithCharactersInString:@"abcdefghijklmnopqrstuvwxyz0123456789"].invertedSet;
	if (lowered.length == 0 || lowered.length > 8 || [lowered rangeOfCharacterFromSet:invalid].location != NSNotFound) {
		return @"bin";
	}
	return lowered;
}

/*! The value types that cross as keyed archives. Decoding admits these and the property-list
	classes their own coders nest, nothing else. */
static NSArray<Class> *NFKWireArchivedClasses(void)
{
	return @[ NFKDetection.class, NFKKeypoint.class, NFKClassification.class, NFKAudioSegment.class,
			  NFKMIDISequence.class, NFKMIDINote.class, NFKMusicBeat.class, NFKQuadrilateral.class,
			  NFKDecisionAnswer.class ];
}

static NSSet<Class> *NFKWireArchiveDecodingClasses(void)
{
	NSMutableSet<Class> *classes = [NSMutableSet setWithArray:NFKWireArchivedClasses()];
	[classes addObjectsFromArray:@[ NSArray.class, NSDictionary.class, NSString.class, NSNumber.class,
									NSData.class, NSDate.class, NSNull.class ]];
	return classes;
}

@interface NFKInferenceWireCoding ()
@property (nonatomic, copy, nullable) NSURL *temporaryDirectory;
@property (nonatomic, strong) NSMutableArray<NSURL *> *writtenFiles;
@end

@implementation NFKInferenceWireCoding

- (instancetype)initWithTemporaryDirectory:(nullable NSURL *)directory
{
	self = [super init];
	if (self != nil) {
		_temporaryDirectory = [directory copy];
		_writtenFiles = [NSMutableArray array];
	}
	return self;
}

- (NSArray<NSURL *> *)temporaryFileURLs
{
	@synchronized (self) {
		return [self.writtenFiles copy];
	}
}

#pragma mark Requests and results

- (nullable id)JSONObjectForRequest:(NFKInferenceRequest *)request error:(NSError * _Nullable *)outError
{
	NSDictionary *inputs = [self JSONDictionaryForDictionary:request.inputs error:outError];
	NSDictionary *parameters = inputs != nil ? [self JSONDictionaryForDictionary:request.parameters error:outError] : nil;
	if (parameters == nil) {
		return nil;
	}
	return @{ @"inputs": inputs, @"parameters": parameters, @"outputModality": @(request.outputModality) };
}

- (nullable NFKInferenceRequest *)requestFromJSONObject:(nullable id)object error:(NSError * _Nullable *)outError
{
	NSDictionary *body = [object isKindOfClass:NSDictionary.class] ? object : nil;
	if (![body[@"inputs"] isKindOfClass:NSDictionary.class]) {
		[self.class setError:outError code:kNFKError_InferenceMissingInput reason:@"the request carries no inputs object"];
		return nil;
	}
	NSDictionary *inputs = [self dictionaryFromJSONDictionary:body[@"inputs"] error:outError];
	id wireParameters = [body[@"parameters"] isKindOfClass:NSDictionary.class] ? body[@"parameters"] : @{};
	NSDictionary *parameters = inputs != nil ? [self dictionaryFromJSONDictionary:wireParameters error:outError] : nil;
	if (parameters == nil) {
		return nil;
	}
	NSNumber *modality = [body[@"outputModality"] isKindOfClass:NSNumber.class] ? body[@"outputModality"] : @(NFKModalityImage);
	return [NFKInferenceRequest requestWithInputs:inputs parameters:parameters outputModality:modality.integerValue];
}

- (nullable id)JSONObjectForResult:(NFKInferenceResult *)result error:(NSError * _Nullable *)outError
{
	NSDictionary *outputs = [self JSONDictionaryForDictionary:result.outputs error:outError];
	return outputs != nil ? @{ @"outputs": outputs } : nil;
}

- (nullable NFKInferenceResult *)resultFromJSONObject:(nullable id)object error:(NSError * _Nullable *)outError
{
	NSDictionary *body = [object isKindOfClass:NSDictionary.class] ? object : nil;
	if (![body[@"outputs"] isKindOfClass:NSDictionary.class]) {
		[self.class setError:outError code:kNFKError_InferenceBackendFailure reason:@"the reply carries no outputs object"];
		return nil;
	}
	NSDictionary *outputs = [self dictionaryFromJSONDictionary:body[@"outputs"] error:outError];
	return outputs != nil ? [NFKInferenceResult resultWithOutputs:outputs] : nil;
}

- (nullable NSDictionary *)JSONDictionaryForDictionary:(NSDictionary *)dictionary error:(NSError * _Nullable *)outError
{
	NSMutableDictionary *encoded = [NSMutableDictionary dictionaryWithCapacity:dictionary.count];
	for (id key in dictionary) {
		if (![key isKindOfClass:NSString.class]) {
			[self.class setError:outError code:kNFKError_InferenceUnsupported
						  reason:[NSString stringWithFormat:@"a dictionary key of class %@ cannot cross the wire; keys are strings", [key class]]];
			return nil;
		}
		id value = [self JSONValueForValue:dictionary[key] key:key error:outError];
		if (value == nil) {
			return nil;
		}
		encoded[key] = value;
	}
	return encoded;
}

- (nullable NSDictionary *)dictionaryFromJSONDictionary:(NSDictionary *)dictionary error:(NSError * _Nullable *)outError
{
	NSMutableDictionary *decoded = [NSMutableDictionary dictionaryWithCapacity:dictionary.count];
	for (NSString *key in dictionary) {
		id value = [self valueFromJSONValue:dictionary[key] key:key error:outError];
		if (value == nil) {
			return nil;
		}
		decoded[key] = value;
	}
	return decoded;
}

#pragma mark Encoding

- (nullable id)JSONValueForValue:(id)value key:(NSString *)key error:(NSError * _Nullable *)outError
{
	if (value == nil || value == NSNull.null || [value isKindOfClass:NSString.class]) {
		return value ?: NSNull.null;
	}
	if ([value isKindOfClass:NSNumber.class]) {
		return [self JSONValueForNumber:value];
	}
	if ([value isKindOfClass:NSArray.class] && NFKWireIsFloatingArray(value)) {
		return [self JSONValueForFloatingArray:value];
	}
	if ([value isKindOfClass:NSArray.class]) {
		NSMutableArray *encoded = [NSMutableArray arrayWithCapacity:[value count]];
		for (id element in value) {
			id item = [self JSONValueForValue:element key:key error:outError];
			if (item == nil) {
				return nil;
			}
			[encoded addObject:item];
		}
		return encoded;
	}
	if ([value isKindOfClass:NSDictionary.class]) {
		NSDictionary *encoded = [self JSONDictionaryForDictionary:value error:outError];
		if (encoded == nil || encoded[NFKWireTag] == nil) {
			return encoded;
		}
		return @{ NFKWireTag: @"dictionary", @"value": encoded };
	}
	if ([value isKindOfClass:NSData.class]) {
		return @{ NFKWireTag: @"data", @"base64": [value base64EncodedStringWithOptions:0] };
	}
	if ([value isKindOfClass:NSDate.class]) {
		return @{ NFKWireTag: @"date", @"seconds": @([value timeIntervalSince1970]) };
	}
	if ([value isKindOfClass:NSURL.class]) {
		return [self JSONValueForURL:value key:key error:outError];
	}
	if ([value isKindOfClass:NFKAudioAsset.class]) {
		return [self JSONValueForAudioAsset:value key:key error:outError];
	}
	if ([value isKindOfClass:NFKVideoAsset.class]) {
		return [self JSONValueForVideoAsset:value key:key error:outError];
	}
	if ([value isKindOfClass:NFKDecisionQuestion.class]) {
		return @{ NFKWireTag: @"decisionQuestion", @"value": [value dictionaryRepresentation] };
	}
	if ([value isKindOfClass:MLMultiArray.class]) {
		return [self JSONValueForMultiArray:value];
	}
	if ([value isKindOfClass:AVAudioPCMBuffer.class]) {
		return [self JSONValueForPCMBuffer:value key:key error:outError];
	}
	for (Class archived in NFKWireArchivedClasses()) {
		if ([value isMemberOfClass:archived]) {
			return [self JSONValueForArchivedValue:value key:key error:outError];
		}
	}
	CFTypeID typeID = CFGetTypeID((__bridge CFTypeRef)value);
	if (typeID == CVPixelBufferGetTypeID()) {
		return [self JSONValueForPixelBuffer:(__bridge CVPixelBufferRef)value key:key error:outError];
	}
	if (typeID == CGImageGetTypeID() || [value conformsToProtocol:@protocol(MTLTexture)]) {
		NSData *png = [NFKImageCoding PNGDataForImage:value];
		if (png == nil) {
			return [self failForKey:key reason:@"the image could not be encoded as PNG" error:outError];
		}
		NSString *tag = typeID == CGImageGetTypeID() ? @"cgImage" : @"image";
		return @{ NFKWireTag: tag, @"png": [png base64EncodedStringWithOptions:0] };
	}
	return [self failForKey:key reason:[NSString stringWithFormat:@"a value of class %@ has no wire form", [value class]] error:outError];
}

// JSON carries integers exactly, and NSJSONSerialization rounds some doubles in the last bit on the
// way through, so a floating value that would not come back identical travels as its 17 significant
// digits, which strtod reads back exactly.
- (id)JSONValueForNumber:(NSNumber *)number
{
	if (!NFKWireIsFloating(number)) {
		return number;
	}
	double value = number.doubleValue;
	if (!isfinite(value)) {
		return @{ NFKWireTag: @"number", @"value": isnan(value) ? @"nan" : (value > 0 ? @"inf" : @"-inf") };
	}
	NSData *JSON = [NSJSONSerialization dataWithJSONObject:@[ number ] options:0 error:NULL];
	NSArray *parsed = JSON != nil ? [NSJSONSerialization JSONObjectWithData:JSON options:0 error:NULL] : nil;
	if ([parsed.firstObject isKindOfClass:NSNumber.class] && [parsed.firstObject doubleValue] == value) {
		return number;
	}
	return @{ NFKWireTag: @"double", @"value": [NSString stringWithFormat:@"%.17g", value] };
}

/*! An array of floating values packed as little-endian IEEE words: float32 when every element is a
	float, float64 otherwise. An embedding crosses exactly and at a third of its decimal size. */
- (id)JSONValueForFloatingArray:(NSArray<NSNumber *> *)array
{
	BOOL singles = YES;
	for (NSNumber *element in array) {
		singles = singles && strcmp(element.objCType, @encode(float)) == 0;
	}
	NSMutableData *packed = [NSMutableData dataWithLength:array.count * (singles ? sizeof(float) : sizeof(double))];
	[array enumerateObjectsUsingBlock:^(NSNumber *element, NSUInteger index, BOOL *stop) {
		if (singles) {
			((float *)packed.mutableBytes)[index] = element.floatValue;
		} else {
			((double *)packed.mutableBytes)[index] = element.doubleValue;
		}
	}];
	return @{ NFKWireTag: singles ? @"float32Array" : @"float64Array", @"base64": [packed base64EncodedStringWithOptions:0] };
}

- (nullable id)JSONValueForURL:(NSURL *)url key:(NSString *)key error:(NSError * _Nullable *)outError
{
	if (!url.isFileURL) {
		return @{ NFKWireTag: @"url", @"value": url.absoluteString };
	}
	NSMutableDictionary *entry = [self JSONEntryForFileURL:url key:key error:outError];
	if (entry == nil) {
		return nil;
	}
	entry[NFKWireTag] = @"file";
	return entry;
}

- (nullable NSMutableDictionary *)JSONEntryForFileURL:(nullable NSURL *)url key:(NSString *)key error:(NSError * _Nullable *)outError
{
	NSMutableDictionary *entry = [NSMutableDictionary dictionary];
	if (url == nil) {
		return entry;
	}
	NSError *readError = nil;
	NSData *bytes = [NSData dataWithContentsOfURL:url options:NSDataReadingMappedIfSafe error:&readError];
	if (bytes == nil) {
		return [self failForKey:key reason:[NSString stringWithFormat:@"the file %@ could not be read: %@",
											url.lastPathComponent, readError.localizedDescription] error:outError];
	}
	entry[@"name"] = url.lastPathComponent;
	entry[@"base64"] = [bytes base64EncodedStringWithOptions:0];
	return entry;
}

- (nullable id)JSONValueForAudioAsset:(NFKAudioAsset *)asset key:(NSString *)key error:(NSError * _Nullable *)outError
{
	NSMutableDictionary *entry = [self JSONEntryForFileURL:asset.fileURL key:key error:outError];
	if (entry == nil) {
		return nil;
	}
	entry[NFKWireTag] = @"audio";
	entry[@"durationSeconds"] = @(asset.durationSeconds);
	entry[@"sampleRate"] = @(asset.sampleRate);
	entry[@"channelCount"] = @(asset.channelCount);
	return entry;
}

- (nullable id)JSONValueForVideoAsset:(NFKVideoAsset *)asset key:(NSString *)key error:(NSError * _Nullable *)outError
{
	NSMutableDictionary *entry = [self JSONEntryForFileURL:asset.fileURL key:key error:outError];
	if (entry == nil) {
		return nil;
	}
	entry[NFKWireTag] = @"video";
	entry[@"durationSeconds"] = @(asset.durationSeconds);
	entry[@"framesPerSecond"] = @(asset.framesPerSecond);
	entry[@"width"] = @(asset.dimensions.width);
	entry[@"height"] = @(asset.dimensions.height);
	return entry;
}

- (nullable id)JSONValueForArchivedValue:(id<NSSecureCoding>)value key:(NSString *)key error:(NSError * _Nullable *)outError
{
	NSError *archiveError = nil;
	NSData *archive = [NSKeyedArchiver archivedDataWithRootObject:value requiringSecureCoding:YES error:&archiveError];
	if (archive == nil) {
		return [self failForKey:key reason:archiveError.localizedDescription ?: @"the value could not be archived" error:outError];
	}
	return @{ NFKWireTag: @"archive", @"base64": [archive base64EncodedStringWithOptions:0] };
}

// Each plane goes with its rows as the buffer holds them, padding included, so any pixel format
// crosses without a table of bytes per pixel; the decoder copies row by row into its own layout.
- (nullable id)JSONValueForPixelBuffer:(CVPixelBufferRef)buffer key:(NSString *)key error:(NSError * _Nullable *)outError
{
	if (CVPixelBufferLockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess) {
		return [self failForKey:key reason:@"the pixel buffer could not be locked for reading" error:outError];
	}
	BOOL planar = CVPixelBufferIsPlanar(buffer);
	size_t planeCount = planar ? CVPixelBufferGetPlaneCount(buffer) : 1;
	NSMutableArray *planes = [NSMutableArray arrayWithCapacity:planeCount];
	for (size_t plane = 0; plane < planeCount; plane++) {
		const void *base = planar ? CVPixelBufferGetBaseAddressOfPlane(buffer, plane) : CVPixelBufferGetBaseAddress(buffer);
		size_t bytesPerRow = planar ? CVPixelBufferGetBytesPerRowOfPlane(buffer, plane) : CVPixelBufferGetBytesPerRow(buffer);
		size_t rows = planar ? CVPixelBufferGetHeightOfPlane(buffer, plane) : CVPixelBufferGetHeight(buffer);
		NSData *bytes = [NSData dataWithBytes:base length:bytesPerRow * rows];
		[planes addObject:@{ @"bytesPerRow": @(bytesPerRow), @"rows": @(rows), @"base64": [bytes base64EncodedStringWithOptions:0] }];
	}
	CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
	return @{ NFKWireTag: @"pixelBuffer",
			  @"format": @(CVPixelBufferGetPixelFormatType(buffer)),
			  @"width": @(CVPixelBufferGetWidth(buffer)),
			  @"height": @(CVPixelBufferGetHeight(buffer)),
			  @"planes": planes };
}

- (id)JSONValueForMultiArray:(MLMultiArray *)array
{
	NSInteger elementSize = (array.dataType & 0xFF) / 8;
	NSMutableData *bytes = [NSMutableData dataWithLength:(NSUInteger)(array.count * elementSize)];
	NFKWireCopyMultiArrayElements(array.dataPointer, array.shape, array.strides, bytes.mutableBytes, elementSize, YES);
	return @{ NFKWireTag: @"multiArray", @"dataType": @(array.dataType), @"shape": array.shape,
			  @"base64": [bytes base64EncodedStringWithOptions:0] };
}

/*! Whether the planes a client sent cover a buffer of these dimensions in this format: each plane's
	rows hold at least the bytes its width needs, there are at least as many as its height needs, and
	its bytes hold them all. The buffer made for the entry is then no larger than what was sent. */
static BOOL NFKWirePlanesCoverDimensions(OSType format, size_t width, size_t height, NSArray *planes)
{
	NSDictionary *description = CFBridgingRelease(CVPixelFormatDescriptionCreateWithPixelFormatType(kCFAllocatorDefault, format));
	if (description == nil) {
		return NO;
	}
	NSArray<NSDictionary *> *layouts = description[(id)kCVPixelFormatPlanes] ?: @[ description ];
	if (layouts.count != planes.count) {
		return NO;
	}
	for (NSUInteger index = 0; index < planes.count; index++) {
		NSDictionary *plane = [planes[index] isKindOfClass:NSDictionary.class] ? planes[index] : @{};
		NSDictionary *layout = layouts[index];
		double bits = [layout[(id)kCVPixelFormatBitsPerBlock] ?: description[(id)kCVPixelFormatBitsPerBlock] doubleValue];
		double blockWidth = MAX([layout[(id)kCVPixelFormatBlockWidth] doubleValue], 1.0);
		double horizontal = MAX([layout[(id)kCVPixelFormatHorizontalSubsampling] doubleValue], 1.0);
		double vertical = MAX([layout[(id)kCVPixelFormatVerticalSubsampling] doubleValue], 1.0);
		double bytesPerRow = [plane[@"bytesPerRow"] isKindOfClass:NSNumber.class] ? [plane[@"bytesPerRow"] doubleValue] : 0;
		double rows = [plane[@"rows"] isKindOfClass:NSNumber.class] ? [plane[@"rows"] doubleValue] : 0;
		double carried = [plane[@"base64"] isKindOfClass:NSString.class] ? floor([plane[@"base64"] length] / 4.0) * 3.0 : 0;
		double neededPerRow = ceil(ceil(width / horizontal) / blockWidth) * bits / 8.0;
		if (bits <= 0 || bytesPerRow < neededPerRow || rows < ceil(height / vertical) || bytesPerRow * rows > carried) {
			return NO;
		}
	}
	return YES;
}

/*! Copies every element between a strided array and a row-major packed buffer, in the direction
	toPacked names. A contiguous array is one copy. */
static void NFKWireCopyMultiArrayElements(void *strided, NSArray<NSNumber *> *shape, NSArray<NSNumber *> *strides,
										  void *packed, NSInteger elementSize, BOOL toPacked)
{
	NSUInteger rank = shape.count;
	NSInteger count = 1;
	NSInteger expectedStride = 1;
	BOOL contiguous = YES;
	for (NSInteger axis = (NSInteger)rank - 1; axis >= 0; axis--) {
		contiguous = contiguous && strides[(NSUInteger)axis].integerValue == expectedStride;
		expectedStride *= shape[(NSUInteger)axis].integerValue;
		count *= shape[(NSUInteger)axis].integerValue;
	}
	if (contiguous) {
		memcpy(toPacked ? packed : strided, toPacked ? strided : packed, (size_t)(count * elementSize));
		return;
	}
	for (NSInteger linear = 0; linear < count; linear++) {
		NSInteger remainder = linear;
		NSInteger offset = 0;
		for (NSInteger axis = (NSInteger)rank - 1; axis >= 0; axis--) {
			NSInteger extent = shape[(NSUInteger)axis].integerValue;
			offset += (remainder % extent) * strides[(NSUInteger)axis].integerValue;
			remainder /= extent;
		}
		char *stridedElement = (char *)strided + offset * elementSize;
		char *packedElement = (char *)packed + linear * elementSize;
		memcpy(toPacked ? packedElement : stridedElement, toPacked ? stridedElement : packedElement, (size_t)elementSize);
	}
}

- (nullable id)JSONValueForPCMBuffer:(AVAudioPCMBuffer *)buffer key:(NSString *)key error:(NSError * _Nullable *)outError
{
	const AudioStreamBasicDescription *description = buffer.format.streamDescription;
	if (description == NULL || description->mBytesPerFrame == 0) {
		return [self failForKey:key reason:@"the PCM buffer's format has no fixed frame size" error:outError];
	}
	const AudioBufferList *list = buffer.audioBufferList;
	NSUInteger length = (NSUInteger)buffer.frameLength * description->mBytesPerFrame;
	NSMutableArray<NSString *> *buffers = [NSMutableArray arrayWithCapacity:list->mNumberBuffers];
	for (UInt32 index = 0; index < list->mNumberBuffers; index++) {
		NSUInteger available = MIN(length, (NSUInteger)list->mBuffers[index].mDataByteSize);
		NSData *bytes = [NSData dataWithBytes:list->mBuffers[index].mData length:available];
		[buffers addObject:[bytes base64EncodedStringWithOptions:0]];
	}
	return @{ NFKWireTag: @"pcmBuffer",
			  @"sampleRate": @(description->mSampleRate),
			  @"formatID": @(description->mFormatID),
			  @"formatFlags": @(description->mFormatFlags),
			  @"bytesPerPacket": @(description->mBytesPerPacket),
			  @"framesPerPacket": @(description->mFramesPerPacket),
			  @"bytesPerFrame": @(description->mBytesPerFrame),
			  @"channelsPerFrame": @(description->mChannelsPerFrame),
			  @"bitsPerChannel": @(description->mBitsPerChannel),
			  @"frameLength": @(buffer.frameLength),
			  @"buffers": buffers };
}

#pragma mark Decoding

- (nullable id)valueFromJSONValue:(id)JSONValue key:(NSString *)key error:(NSError * _Nullable *)outError
{
	if ([JSONValue isKindOfClass:NSArray.class]) {
		NSMutableArray *decoded = [NSMutableArray arrayWithCapacity:[JSONValue count]];
		for (id element in JSONValue) {
			id item = [self valueFromJSONValue:element key:key error:outError];
			if (item == nil) {
				return nil;
			}
			[decoded addObject:item];
		}
		return decoded;
	}
	if (![JSONValue isKindOfClass:NSDictionary.class]) {
		return JSONValue;
	}
	NSDictionary *entry = JSONValue;
	NSString *tag = [entry[NFKWireTag] isKindOfClass:NSString.class] ? entry[NFKWireTag] : nil;
	if (tag == nil) {
		return [self dictionaryFromJSONDictionary:entry error:outError];
	}
	if ([tag isEqualToString:@"dictionary"] && [entry[@"value"] isKindOfClass:NSDictionary.class]) {
		return [self dictionaryFromJSONDictionary:entry[@"value"] error:outError];
	}
	if ([tag isEqualToString:@"number"]) {
		NSDictionary<NSString *, NSNumber *> *numbers = @{ @"nan": @(NAN), @"inf": @(INFINITY), @"-inf": @(-INFINITY) };
		return numbers[entry[@"value"]] ?: [self failForKey:key reason:@"an unknown non-finite number" error:outError];
	}
	if ([tag isEqualToString:@"double"] && [entry[@"value"] isKindOfClass:NSString.class]) {
		return @(strtod([entry[@"value"] UTF8String], NULL));
	}
	if ([tag isEqualToString:@"url"] && [entry[@"value"] isKindOfClass:NSString.class]) {
		return [NSURL URLWithString:entry[@"value"]] ?: [self failForKey:key reason:@"a malformed URL" error:outError];
	}
	if ([tag isEqualToString:@"date"] && [entry[@"seconds"] isKindOfClass:NSNumber.class]) {
		return [NSDate dateWithTimeIntervalSince1970:[entry[@"seconds"] doubleValue]];
	}
	if ([tag isEqualToString:@"decisionQuestion"] && [entry[@"value"] isKindOfClass:NSDictionary.class]) {
		return [NFKDecisionQuestion questionWithDictionary:entry[@"value"]]
			?: [self failForKey:key reason:@"a malformed decision question" error:outError];
	}
	NSData *bytes = [self bytesIn:entry forKey:@"base64"];
	if ([tag isEqualToString:@"data"] && bytes != nil) {
		return bytes;
	}
	if (([tag isEqualToString:@"float32Array"] || [tag isEqualToString:@"float64Array"]) && bytes != nil) {
		return [self floatingArrayFromBytes:bytes singles:[tag isEqualToString:@"float32Array"] key:key error:outError];
	}
	if ([tag isEqualToString:@"file"] || [tag isEqualToString:@"audio"] || [tag isEqualToString:@"video"]) {
		return [self fileValueForTag:tag entry:entry bytes:bytes key:key error:outError];
	}
	if ([tag isEqualToString:@"archive"] && bytes != nil) {
		NSError *archiveError = nil;
		id value = [NSKeyedUnarchiver unarchivedObjectOfClasses:NFKWireArchiveDecodingClasses() fromData:bytes error:&archiveError];
		return value ?: [self failForKey:key reason:archiveError.localizedDescription ?: @"the archive could not be read" error:outError];
	}
	if ([tag isEqualToString:@"pixelBuffer"]) {
		return [self pixelBufferFromEntry:entry key:key error:outError];
	}
	NSData *png = [self bytesIn:entry forKey:@"png"];
	if ([tag isEqualToString:@"cgImage"] && png != nil) {
		return [self CGImageFromData:png key:key error:outError];
	}
	if ([tag isEqualToString:@"image"] && png != nil) {
		CVPixelBufferRef pixelBuffer = [NFKImageCoding pixelBufferWithImageData:png];
		if (pixelBuffer == NULL) {
			return [self failForKey:key reason:@"the image could not be decoded" error:outError];
		}
		return (__bridge_transfer id)pixelBuffer;
	}
	if ([tag isEqualToString:@"multiArray"] && bytes != nil) {
		return [self multiArrayFromEntry:entry bytes:bytes key:key error:outError];
	}
	if ([tag isEqualToString:@"pcmBuffer"]) {
		return [self PCMBufferFromEntry:entry key:key error:outError];
	}
	return [self failForKey:key reason:[NSString stringWithFormat:@"the wire tag \"%@\" is unknown or malformed", tag] error:outError];
}

- (nullable NSData *)bytesIn:(NSDictionary *)entry forKey:(NSString *)field
{
	NSString *encoded = [entry[field] isKindOfClass:NSString.class] ? entry[field] : nil;
	return encoded != nil ? [[NSData alloc] initWithBase64EncodedString:encoded options:0] : nil;
}

- (double)doubleIn:(NSDictionary *)entry forKey:(NSString *)field
{
	return [entry[field] isKindOfClass:NSNumber.class] ? [entry[field] doubleValue] : 0.0;
}

- (nullable id)floatingArrayFromBytes:(NSData *)bytes singles:(BOOL)singles key:(NSString *)key error:(NSError * _Nullable *)outError
{
	NSUInteger width = singles ? sizeof(float) : sizeof(double);
	if (bytes.length % width != 0) {
		return [self failForKey:key reason:@"a packed number array has a partial element" error:outError];
	}
	NSMutableArray<NSNumber *> *numbers = [NSMutableArray arrayWithCapacity:bytes.length / width];
	for (NSUInteger index = 0; index < bytes.length / width; index++) {
		[numbers addObject:singles ? @(((const float *)bytes.bytes)[index]) : @(((const double *)bytes.bytes)[index])];
	}
	return numbers;
}

- (nullable id)fileValueForTag:(NSString *)tag entry:(NSDictionary *)entry bytes:(nullable NSData *)bytes key:(NSString *)key error:(NSError * _Nullable *)outError
{
	NSURL *fileURL = nil;
	if (bytes != nil) {
		NSString *name = [entry[@"name"] isKindOfClass:NSString.class] ? entry[@"name"] : @"";
		fileURL = NFKRemoteWriteMediaFile(bytes, @"served", NFKServedFileExtension(name.pathExtension), self.temporaryDirectory, outError);
		if (fileURL == nil) {
			return nil;
		}
		@synchronized (self) {
			[self.writtenFiles addObject:fileURL];
		}
	}
	if ([tag isEqualToString:@"file"]) {
		return fileURL ?: [self failForKey:key reason:@"a file entry carries no bytes" error:outError];
	}
	if ([tag isEqualToString:@"audio"]) {
		return [NFKAudioAsset audioAssetWithFileURL:fileURL
									durationSeconds:[self doubleIn:entry forKey:@"durationSeconds"]
										 sampleRate:[self doubleIn:entry forKey:@"sampleRate"]
									   channelCount:(NSInteger)[self doubleIn:entry forKey:@"channelCount"]];
	}
	return [NFKVideoAsset videoAssetWithFileURL:fileURL
								durationSeconds:[self doubleIn:entry forKey:@"durationSeconds"]
								framesPerSecond:[self doubleIn:entry forKey:@"framesPerSecond"]
									 dimensions:CGSizeMake([self doubleIn:entry forKey:@"width"], [self doubleIn:entry forKey:@"height"])];
}

- (nullable id)CGImageFromData:(NSData *)data key:(NSString *)key error:(NSError * _Nullable *)outError
{
	CGImageSourceRef source = CGImageSourceCreateWithData((__bridge CFDataRef)data, NULL);
	CGImageRef image = source != NULL ? CGImageSourceCreateImageAtIndex(source, 0, NULL) : NULL;
	if (source != NULL) {
		CFRelease(source);
	}
	if (image == NULL) {
		return [self failForKey:key reason:@"the image could not be decoded" error:outError];
	}
	return (__bridge_transfer id)image;
}

- (nullable id)pixelBufferFromEntry:(NSDictionary *)entry key:(NSString *)key error:(NSError * _Nullable *)outError
{
	NSArray *planes = [entry[@"planes"] isKindOfClass:NSArray.class] ? entry[@"planes"] : nil;
	size_t width = (size_t)[self doubleIn:entry forKey:@"width"];
	size_t height = (size_t)[self doubleIn:entry forKey:@"height"];
	OSType format = (OSType)[self doubleIn:entry forKey:@"format"];
	if (planes.count == 0 || width == 0 || height == 0 || !NFKWirePlanesCoverDimensions(format, width, height, planes)) {
		return [self failForKey:key reason:@"a malformed pixel buffer" error:outError];
	}
	NSDictionary *attributes = @{ (id)kCVPixelBufferIOSurfacePropertiesKey: @{},
								  (id)kCVPixelBufferMetalCompatibilityKey: @YES };
	CVPixelBufferRef buffer = NULL;
	if (CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, (__bridge CFDictionaryRef)attributes, &buffer) != kCVReturnSuccess
		&& CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, NULL, &buffer) != kCVReturnSuccess) {
		return [self failForKey:key reason:[NSString stringWithFormat:@"no pixel buffer of format %u can be made here", (unsigned)format] error:outError];
	}
	id owned = (__bridge_transfer id)buffer;
	BOOL planar = CVPixelBufferIsPlanar(buffer);
	size_t planeCount = planar ? CVPixelBufferGetPlaneCount(buffer) : 1;
	if (planeCount != planes.count) {
		return [self failForKey:key reason:@"the pixel buffer's plane count does not match its format" error:outError];
	}
	CVPixelBufferLockBaseAddress(buffer, 0);
	for (size_t plane = 0; plane < planeCount; plane++) {
		NSDictionary *wirePlane = [planes[plane] isKindOfClass:NSDictionary.class] ? planes[plane] : @{};
		NSData *bytes = [self bytesIn:wirePlane forKey:@"base64"];
		size_t sourceBytesPerRow = (size_t)[self doubleIn:wirePlane forKey:@"bytesPerRow"];
		size_t sourceRows = (size_t)[self doubleIn:wirePlane forKey:@"rows"];
		char *destination = planar ? CVPixelBufferGetBaseAddressOfPlane(buffer, plane) : CVPixelBufferGetBaseAddress(buffer);
		size_t destinationBytesPerRow = planar ? CVPixelBufferGetBytesPerRowOfPlane(buffer, plane) : CVPixelBufferGetBytesPerRow(buffer);
		size_t destinationRows = planar ? CVPixelBufferGetHeightOfPlane(buffer, plane) : CVPixelBufferGetHeight(buffer);
		if (bytes == nil || bytes.length < sourceBytesPerRow * sourceRows || sourceRows != destinationRows) {
			CVPixelBufferUnlockBaseAddress(buffer, 0);
			return [self failForKey:key reason:@"a pixel buffer plane is short or misshapen" error:outError];
		}
		size_t copied = MIN(sourceBytesPerRow, destinationBytesPerRow);
		for (size_t row = 0; row < destinationRows; row++) {
			memcpy(destination + row * destinationBytesPerRow, (const char *)bytes.bytes + row * sourceBytesPerRow, copied);
		}
	}
	CVPixelBufferUnlockBaseAddress(buffer, 0);
	return owned;
}

- (nullable id)multiArrayFromEntry:(NSDictionary *)entry bytes:(NSData *)bytes key:(NSString *)key error:(NSError * _Nullable *)outError
{
	NSArray<NSNumber *> *shape = [entry[@"shape"] isKindOfClass:NSArray.class] ? entry[@"shape"] : nil;
	MLMultiArrayDataType dataType = (MLMultiArrayDataType)[self doubleIn:entry forKey:@"dataType"];
	NSInteger elementSize = (dataType & 0xFF) / 8;
	// The shape must account for exactly the bytes sent before anything is allocated for it.
	unsigned long long expected = elementSize > 0 && shape.count > 0 ? (unsigned long long)elementSize : 0;
	for (NSNumber *extent in shape) {
		BOOL valid = [extent isKindOfClass:NSNumber.class] && extent.longLongValue > 0 && expected <= bytes.length;
		expected = valid ? expected * (unsigned long long)extent.longLongValue : 0;
	}
	if (expected == 0 || expected != bytes.length) {
		return [self failForKey:key reason:@"the multi-array's shape does not match its bytes" error:outError];
	}
	NSError *createError = nil;
	MLMultiArray *array = [[MLMultiArray alloc] initWithShape:shape dataType:dataType error:&createError];
	if (array == nil || (NSInteger)bytes.length != array.count * elementSize) {
		return [self failForKey:key reason:createError.localizedDescription ?: @"a malformed multi-array" error:outError];
	}
	NFKWireCopyMultiArrayElements(array.dataPointer, array.shape, array.strides, (void *)bytes.bytes, elementSize, NO);
	return array;
}

- (nullable id)PCMBufferFromEntry:(NSDictionary *)entry key:(NSString *)key error:(NSError * _Nullable *)outError
{
	AudioStreamBasicDescription description = {0};
	description.mSampleRate = [self doubleIn:entry forKey:@"sampleRate"];
	description.mFormatID = (AudioFormatID)[self doubleIn:entry forKey:@"formatID"];
	description.mFormatFlags = (AudioFormatFlags)[self doubleIn:entry forKey:@"formatFlags"];
	description.mBytesPerPacket = (UInt32)[self doubleIn:entry forKey:@"bytesPerPacket"];
	description.mFramesPerPacket = (UInt32)[self doubleIn:entry forKey:@"framesPerPacket"];
	description.mBytesPerFrame = (UInt32)[self doubleIn:entry forKey:@"bytesPerFrame"];
	description.mChannelsPerFrame = (UInt32)[self doubleIn:entry forKey:@"channelsPerFrame"];
	description.mBitsPerChannel = (UInt32)[self doubleIn:entry forKey:@"bitsPerChannel"];
	// More than two channels needs a layout for AVAudioFormat to accept the description.
	AVAudioChannelLayout *layout = description.mChannelsPerFrame > 2
		? [AVAudioChannelLayout layoutWithLayoutTag:kAudioChannelLayoutTag_DiscreteInOrder | description.mChannelsPerFrame] : nil;
	AVAudioFormat *format = [[AVAudioFormat alloc] initWithStreamDescription:&description channelLayout:layout];
	AVAudioFrameCount frames = (AVAudioFrameCount)[self doubleIn:entry forKey:@"frameLength"];
	NSMutableArray<NSData *> *samples = [NSMutableArray array];
	for (id encoded in [entry[@"buffers"] isKindOfClass:NSArray.class] ? entry[@"buffers"] : @[]) {
		NSData *bytes = [encoded isKindOfClass:NSString.class] ? [[NSData alloc] initWithBase64EncodedString:encoded options:0] : nil;
		// Every buffer holds exactly the frames named, so the frame count cannot ask for more memory
		// than the body carried.
		if (bytes == nil || bytes.length != (NSUInteger)frames * description.mBytesPerFrame) {
			return [self failForKey:key reason:@"a PCM buffer's samples do not match its frame count" error:outError];
		}
		[samples addObject:bytes];
	}
	AVAudioPCMBuffer *buffer = format != nil ? [[AVAudioPCMBuffer alloc] initWithPCMFormat:format frameCapacity:MAX(frames, 1)] : nil;
	if (buffer == nil || samples.count != buffer.audioBufferList->mNumberBuffers) {
		return [self failForKey:key reason:@"a malformed PCM buffer" error:outError];
	}
	AudioBufferList *list = buffer.mutableAudioBufferList;
	for (UInt32 index = 0; index < list->mNumberBuffers; index++) {
		if (samples[index].length > list->mBuffers[index].mDataByteSize) {
			return [self failForKey:key reason:@"a PCM buffer's samples are malformed" error:outError];
		}
		memcpy(list->mBuffers[index].mData, samples[index].bytes, samples[index].length);
	}
	buffer.frameLength = frames;
	return buffer;
}

#pragma mark Errors

+ (NSDictionary<NSString *, id> *)JSONObjectForError:(NSError *)error
{
	NSMutableDictionary<NSString *, id> *userInfo = [NSMutableDictionary dictionary];
	[error.userInfo enumerateKeysAndObjectsUsingBlock:^(NSString *key, id value, BOOL *stop) {
		BOOL finiteNumber = [value isKindOfClass:NSNumber.class] && isfinite([value doubleValue]);
		if (![key isEqualToString:NSLocalizedDescriptionKey] && ([value isKindOfClass:NSString.class] || finiteNumber)) {
			userInfo[key] = value;
		}
	}];
	BOOL ours = [error.domain isEqualToString:NFKInferenceErrorDomain];
	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
	body[@"message"] = error.localizedDescription ?: @"the run failed";
	body[@"type"] = ours ? [self typeForCode:error.code] : @"server_error";
	body[@"code"] = ours ? [self codeNameForCode:error.code] : nil;
	body[@"inferkit_domain"] = error.domain;
	body[@"inferkit_code"] = @(error.code);
	body[@"inferkit_user_info"] = userInfo;
	return @{ @"error": body };
}

+ (nullable NSError *)errorFromJSONObject:(nullable id)object
{
	NSDictionary *error = [object isKindOfClass:NSDictionary.class] && [object[@"error"] isKindOfClass:NSDictionary.class] ? object[@"error"] : nil;
	NSString *domain = [error[@"inferkit_domain"] isKindOfClass:NSString.class] ? error[@"inferkit_domain"] : nil;
	NSNumber *code = [error[@"inferkit_code"] isKindOfClass:NSNumber.class] ? error[@"inferkit_code"] : nil;
	if (domain == nil || code == nil) {
		return nil;
	}
	NSMutableDictionary *userInfo = [NSMutableDictionary dictionary];
	if ([error[@"inferkit_user_info"] isKindOfClass:NSDictionary.class]) {
		[userInfo addEntriesFromDictionary:error[@"inferkit_user_info"]];
	}
	userInfo[NSLocalizedDescriptionKey] = [error[@"message"] isKindOfClass:NSString.class] ? error[@"message"] : @"the served run failed";
	return [NSError errorWithDomain:domain code:code.integerValue userInfo:userInfo];
}

+ (NSString *)typeForCode:(NSInteger)code
{
	switch (code) {
		case kNFKError_InferenceMissingInput:
		case kNFKError_InferenceUnsupported:
		case kNFKError_InferenceRefused:
			return @"invalid_request_error";
		case kNFKError_InferenceRateLimited:
			return @"rate_limit_error";
		default:
			return @"server_error";
	}
}

+ (nullable NSString *)codeNameForCode:(NSInteger)code
{
	NSDictionary<NSNumber *, NSString *> *names = @{ @(kNFKError_InferenceNotReady): @"model_not_ready",
													 @(kNFKError_InferenceMissingInput): @"missing_input",
													 @(kNFKError_InferenceBackendFailure): @"backend_failure",
													 @(kNFKError_InferenceUnsupported): @"unsupported",
													 @(kNFKError_RemoteUnreachable): @"upstream_unreachable",
													 @(kNFKError_InferenceRefused): @"refused",
													 @(kNFKError_InferenceRateLimited): @"rate_limited" };
	return names[@(code)];
}

+ (BOOL)setError:(NSError * _Nullable *)outError code:(NSInteger)code reason:(NSString *)reason
{
	if (outError != NULL) {
		*outError = [NSError errorWithDomain:NFKInferenceErrorDomain code:code userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return NO;
}

- (nullable id)failForKey:(NSString *)key reason:(NSString *)reason error:(NSError * _Nullable *)outError
{
	[self.class setError:outError code:kNFKError_InferenceUnsupported
				  reason:[NSString stringWithFormat:@"the value under \"%@\" cannot cross the wire: %@", key, reason]];
	return nil;
}

@end
