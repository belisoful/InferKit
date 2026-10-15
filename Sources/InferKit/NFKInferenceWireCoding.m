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
#import <InferKit/NFKToolLoop.h>
#import <InferKit/NFKErrors.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>

static NSString * const NFKWireTag = @"$nfk";

static const NSUInteger NFKWireDefaultInlineByteLimit = 64u * 1024u;

/*! The alignment of each plane or buffer inside an attachment, so a wrapped plane suits vector code. */
static const NSUInteger NFKWireAttachmentAlignment = 64;

static NSUInteger NFKWireAlign(NSUInteger offset)
{
	return (offset + NFKWireAttachmentAlignment - 1) / NFKWireAttachmentAlignment * NFKWireAttachmentAlignment;
}

/*! Whether value is a whole number a JSON double carries exactly, 0 through 2^53, stored in outValue. */
static BOOL NFKWireWholeNumber(id value, unsigned long long *outValue)
{
	if (![value isKindOfClass:NSNumber.class] || CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID()) {
		return NO;
	}
	double number = [value doubleValue];
	if (!isfinite(number) || number < 0 || number > 9007199254740992.0 || floor(number) != number) {
		return NO;
	}
	*outValue = (unsigned long long)number;
	return YES;
}

/*! The first length bytes of data, sharing its storage; the slice keeps data alive. */
static NSData *NFKWireLeadingBytes(NSData *data, NSUInteger length)
{
	if (length == data.length) {
		return data;
	}
	if (length == 0) {
		return [NSData data];
	}
	return [[NSData alloc] initWithBytesNoCopy:(void *)data.bytes length:length deallocator:^(void *bytes, NSUInteger sliceLength) {
		(void)data;
	}];
}

/*! Releases the data an attachment-wrapped pixel buffer holds. CoreVideo calls it only for a buffer it
	made, so a failed creation releases the data itself. */
static void NFKWireReleaseAttachedBytes(void *releaseRefCon, const void *baseAddress)
{
	CFRelease(releaseRefCon);
}

static void NFKWireReleaseAttachedPlanes(void *releaseRefCon, const void *dataPtr, size_t dataSize, size_t numberOfPlanes,
										 const void * _Nullable planeAddresses[_Nullable])
{
	CFRelease(releaseRefCon);
}

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
		_inlineByteLimit = NFKWireDefaultInlineByteLimit;
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
		return [self JSONValueForFloatingArray:value key:key error:outError];
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
	if ([value isKindOfClass:NSData.class] && [self attachesLength:[value length]]) {
		NSData *data = value;
		return [self attachmentEntryAs:@"data" length:data.length fields:@{} key:key error:outError fill:^(uint8_t *region) {
			memcpy(region, data.bytes, data.length);
		}];
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
	if ([value isKindOfClass:NFKToolResult.class]) {
		return [self JSONValueForToolResult:value key:key error:outError];
	}
	if ([value isKindOfClass:NFKToolInvocation.class]) {
		return [self JSONValueForToolInvocation:value key:key error:outError];
	}
	if ([value isKindOfClass:MLMultiArray.class]) {
		return [self JSONValueForMultiArray:value key:key error:outError];
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
		if ([self attachesLength:png.length]) {
			return [self attachmentEntryAs:tag length:png.length fields:@{} key:key error:outError fill:^(uint8_t *region) {
				memcpy(region, png.bytes, png.length);
			}];
		}
		return @{ NFKWireTag: tag, @"png": [png base64EncodedStringWithOptions:0] };
	}
	return [self failForKey:key reason:[NSString stringWithFormat:@"a value of class %@ has no wire form", [value class]] error:outError];
}

// A tool's result crosses with its outputs encoded like any other values, so the audio a tool made
// arrives as the audio file it was.
- (nullable id)JSONValueForToolResult:(NFKToolResult *)result key:(NSString *)key error:(NSError * _Nullable *)outError
{
	NSDictionary *outputs = [self JSONDictionaryForDictionary:result.outputs error:outError];
	if (outputs == nil) {
		return nil;
	}
	NSMutableArray<NSString *> *images = [NSMutableArray arrayWithCapacity:result.imagePNGs.count];
	for (NSData *png in result.imagePNGs) {
		[images addObject:[png base64EncodedStringWithOptions:0]];
	}
	return @{ NFKWireTag: @"toolResult", @"text": result.text, @"isError": @(result.isError), @"images": images, @"outputs": outputs };
}

- (nullable id)JSONValueForToolInvocation:(NFKToolInvocation *)invocation key:(NSString *)key error:(NSError * _Nullable *)outError
{
	id result = [self JSONValueForToolResult:invocation.result key:key error:outError];
	NSDictionary *arguments = [self JSONDictionaryForDictionary:invocation.arguments error:outError];
	if (result == nil || arguments == nil) {
		return nil;
	}
	return @{ NFKWireTag: @"toolInvocation", @"callIdentifier": invocation.callIdentifier, @"toolName": invocation.toolName,
			  @"arguments": arguments, @"durationSeconds": @(invocation.durationSeconds), @"result": result };
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
- (nullable id)JSONValueForFloatingArray:(NSArray<NSNumber *> *)array key:(NSString *)key error:(NSError * _Nullable *)outError
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
	NSString *tag = singles ? @"float32Array" : @"float64Array";
	if ([self attachesLength:packed.length]) {
		return [self attachmentEntryAs:tag length:packed.length fields:@{ @"count": @(array.count) } key:key error:outError
								  fill:^(uint8_t *region) {
			memcpy(region, packed.bytes, packed.length);
		}];
	}
	return @{ NFKWireTag: tag, @"base64": [packed base64EncodedStringWithOptions:0] };
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
	if (self.attachmentSink != nil) {
		return [self referenceEntryForFileURL:url key:key error:outError];
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

/*! A file in the attachment mode: its path and name, and the index of an open descriptor when the sink
	attaches one. */
- (nullable NSMutableDictionary *)referenceEntryForFileURL:(NSURL *)url key:(NSString *)key error:(NSError * _Nullable *)outError
{
	NSMutableDictionary *entry = [NSMutableDictionary dictionary];
	entry[@"name"] = url.lastPathComponent;
	entry[@"path"] = url.URLByStandardizingPath.path;
	if (![self.attachmentSink respondsToSelector:@selector(attachFileAtURL:index:error:)]) {
		return entry;
	}
	NSInteger index = -1;
	NSError *attachError = nil;
	if (![self.attachmentSink attachFileAtURL:url index:&index error:&attachError]) {
		return [self failForKey:key reason:[NSString stringWithFormat:@"the file %@ could not be attached: %@",
											url.lastPathComponent, attachError.localizedDescription] error:outError];
	}
	if (index >= 0) {
		entry[@"index"] = @(index);
	}
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
	if (self.attachmentSink != nil) {
		NSError *attachError = nil;
		id attached = [self attachedEntryForLockedPixelBuffer:buffer planeCount:planeCount key:key error:&attachError];
		if (attached != nil || attachError != nil) {
			CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
			if (attached == nil && outError != NULL) {
				*outError = attachError;
			}
			return attached;
		}
	}
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

/*! A locked pixel buffer as one attachment holding every plane at an aligned offset, rows padded as the
	buffer holds them, or nil with no error when its bytes fit inline. An IOSurface-backed buffer is
	copied the same way. */
- (nullable id)attachedEntryForLockedPixelBuffer:(CVPixelBufferRef)buffer planeCount:(size_t)planeCount
											 key:(NSString *)key error:(NSError * _Nullable *)outError
{
	BOOL planar = CVPixelBufferIsPlanar(buffer);
	NSMutableArray<NSDictionary *> *planes = [NSMutableArray arrayWithCapacity:planeCount];
	NSUInteger length = 0;
	NSUInteger payload = 0;
	for (size_t plane = 0; plane < planeCount; plane++) {
		size_t bytesPerRow = planar ? CVPixelBufferGetBytesPerRowOfPlane(buffer, plane) : CVPixelBufferGetBytesPerRow(buffer);
		size_t rows = planar ? CVPixelBufferGetHeightOfPlane(buffer, plane) : CVPixelBufferGetHeight(buffer);
		length = NFKWireAlign(length);
		[planes addObject:@{ @"offset": @(length), @"bytesPerRow": @(bytesPerRow), @"rows": @(rows) }];
		length += bytesPerRow * rows;
		payload += bytesPerRow * rows;
	}
	if (![self attachesLength:payload]) {
		return nil;
	}
	NSDictionary *fields = @{ @"format": @(CVPixelBufferGetPixelFormatType(buffer)),
							  @"width": @(CVPixelBufferGetWidth(buffer)),
							  @"height": @(CVPixelBufferGetHeight(buffer)),
							  @"planes": planes };
	return [self attachmentEntryAs:@"pixelBuffer" length:length fields:fields key:key error:outError fill:^(uint8_t *region) {
		for (size_t plane = 0; plane < planeCount; plane++) {
			const void *base = planar ? CVPixelBufferGetBaseAddressOfPlane(buffer, plane) : CVPixelBufferGetBaseAddress(buffer);
			NSDictionary *layout = planes[plane];
			memcpy(region + [layout[@"offset"] unsignedLongValue], base,
				   [layout[@"bytesPerRow"] unsignedLongValue] * [layout[@"rows"] unsignedLongValue]);
		}
	}];
}

- (nullable id)JSONValueForMultiArray:(MLMultiArray *)array key:(NSString *)key error:(NSError * _Nullable *)outError
{
	NSInteger elementSize = (array.dataType & 0xFF) / 8;
	NSUInteger length = (NSUInteger)(array.count * elementSize);
	if ([self attachesLength:length]) {
		return [self attachmentEntryAs:@"multiArray" length:length fields:@{ @"dataType": @(array.dataType), @"shape": array.shape }
								   key:key error:outError fill:^(uint8_t *region) {
			NFKWireCopyMultiArrayElements(array.dataPointer, array.shape, array.strides, region, elementSize, YES);
		}];
	}
	NSMutableData *bytes = [NSMutableData dataWithLength:length];
	NFKWireCopyMultiArrayElements(array.dataPointer, array.shape, array.strides, bytes.mutableBytes, elementSize, YES);
	return @{ NFKWireTag: @"multiArray", @"dataType": @(array.dataType), @"shape": array.shape,
			  @"base64": [bytes base64EncodedStringWithOptions:0] };
}

/*! Whether the planes a client sent cover a buffer of these dimensions in this format: each plane's
	rows hold at least the bytes its width needs, there are at least as many as its height needs, and
	its bytes hold them all. The buffer made for the entry is then no larger than what was sent. */
static BOOL NFKWirePlanesCoverDimensions(OSType format, size_t width, size_t height, NSArray *planes)
{
	return NFKWirePlaneGeometryCovers(format, width, height, planes, ^double(NSDictionary *plane) {
		return [plane[@"base64"] isKindOfClass:NSString.class] ? floor([plane[@"base64"] length] / 4.0) * 3.0 : 0;
	});
}

/*! The check NFKWirePlanesCoverDimensions makes, with carried giving the bytes each plane can reach. */
static BOOL NFKWirePlaneGeometryCovers(OSType format, size_t width, size_t height, NSArray *planes,
									   double (NS_NOESCAPE ^carriedBytes)(NSDictionary *plane))
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
		double carried = carriedBytes(plane);
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
	if (self.attachmentSink != nil) {
		NSError *attachError = nil;
		id attached = [self attachedEntryForPCMBuffer:buffer frameBytes:length key:key error:&attachError];
		if (attached != nil || attachError != nil) {
			if (attached == nil && outError != NULL) {
				*outError = attachError;
			}
			return attached;
		}
	}
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

/*! The stream description fields both PCM forms carry. */
- (NSMutableDictionary *)fieldsForPCMBuffer:(AVAudioPCMBuffer *)buffer
{
	const AudioStreamBasicDescription *description = buffer.format.streamDescription;
	return [@{ @"sampleRate": @(description->mSampleRate),
			   @"formatID": @(description->mFormatID),
			   @"formatFlags": @(description->mFormatFlags),
			   @"bytesPerPacket": @(description->mBytesPerPacket),
			   @"framesPerPacket": @(description->mFramesPerPacket),
			   @"bytesPerFrame": @(description->mBytesPerFrame),
			   @"channelsPerFrame": @(description->mChannelsPerFrame),
			   @"bitsPerChannel": @(description->mBitsPerChannel),
			   @"frameLength": @(buffer.frameLength) } mutableCopy];
}

/*! A PCM buffer as one attachment holding each audio buffer's frames at an aligned offset, or nil with no
	error when its samples fit inline. */
- (nullable id)attachedEntryForPCMBuffer:(AVAudioPCMBuffer *)buffer frameBytes:(NSUInteger)frameBytes
									 key:(NSString *)key error:(NSError * _Nullable *)outError
{
	const AudioBufferList *list = buffer.audioBufferList;
	NSMutableArray<NSDictionary *> *layouts = [NSMutableArray arrayWithCapacity:list->mNumberBuffers];
	NSUInteger length = 0;
	NSUInteger payload = 0;
	for (UInt32 index = 0; index < list->mNumberBuffers; index++) {
		NSUInteger available = MIN(frameBytes, (NSUInteger)list->mBuffers[index].mDataByteSize);
		length = NFKWireAlign(length);
		[layouts addObject:@{ @"offset": @(length), @"length": @(available) }];
		length += available;
		payload += available;
	}
	if (![self attachesLength:payload]) {
		return nil;
	}
	NSMutableDictionary *fields = [self fieldsForPCMBuffer:buffer];
	fields[@"buffers"] = layouts;
	return [self attachmentEntryAs:@"pcm" length:length fields:fields key:key error:outError fill:^(uint8_t *region) {
		for (UInt32 index = 0; index < list->mNumberBuffers; index++) {
			memcpy(region + [layouts[index][@"offset"] unsignedLongValue], list->mBuffers[index].mData,
				   [layouts[index][@"length"] unsignedLongValue]);
		}
	}];
}

#pragma mark Attachments

- (BOOL)attachesLength:(NSUInteger)length
{
	return self.attachmentSink != nil && length > self.inlineByteLimit;
}

/*! An attachment entry: a region of length bytes that fill writes, tagged "buffer" with kind under "as"
	and fields beside it. */
- (nullable NSDictionary *)attachmentEntryAs:(NSString *)kind length:(NSUInteger)length fields:(NSDictionary *)fields
										 key:(NSString *)key error:(NSError * _Nullable *)outError
										fill:(void (NS_NOESCAPE ^)(uint8_t *region))fill
{
	NSInteger index = -1;
	NSError *sinkError = nil;
	uint8_t *region = [self.attachmentSink attachmentRegionOfLength:length index:&index error:&sinkError];
	if (region == NULL || index < 0) {
		return [self failForKey:key reason:sinkError.localizedDescription ?: @"no attachment region could be made" error:outError];
	}
	fill(region);
	NSMutableDictionary *entry = [NSMutableDictionary dictionaryWithDictionary:fields];
	entry[NFKWireTag] = @"buffer";
	entry[@"index"] = @(index);
	entry[@"as"] = kind;
	entry[@"length"] = @(length);
	return entry;
}

- (nullable id)attachedValueFromEntry:(NSDictionary *)entry key:(NSString *)key error:(NSError * _Nullable *)outError
{
	if (self.attachmentSource == nil) {
		return [self failForKey:key reason:@"an attachment arrived outside the attachment mode" error:outError];
	}
	NSNumber *index = [entry[@"index"] isKindOfClass:NSNumber.class] ? entry[@"index"] : nil;
	NSString *kind = [entry[@"as"] isKindOfClass:NSString.class] ? entry[@"as"] : nil;
	if (index == nil || index.integerValue < 0 || kind == nil) {
		return [self failForKey:key reason:@"a malformed attachment" error:outError];
	}
	unsigned long long length = 0;
	if (!NFKWireWholeNumber(entry[@"length"], &length)) {
		return [self failForKey:key reason:@"an attachment names no length" error:outError];
	}
	NSError *sourceError = nil;
	NSData *region = [self.attachmentSource attachmentDataAtIndex:index.integerValue error:&sourceError];
	if (region == nil) {
		return [self failForKey:key reason:sourceError.localizedDescription ?: @"the attachment is missing" error:outError];
	}
	// The region is at least the attachment: a shared-memory object is whole pages. The entry's length is
	// the attachment's, so every reader below sees exactly those bytes and nothing past them.
	if (length > region.length) {
		return [self failForKey:key reason:[NSString stringWithFormat:@"the attachment names %llu bytes and its region holds %lu",
											length, (unsigned long)region.length] error:outError];
	}
	NSData *data = NFKWireLeadingBytes(region, (NSUInteger)length);
	if ([kind isEqualToString:@"data"]) {
		return data;
	}
	if ([kind isEqualToString:@"float32Array"] || [kind isEqualToString:@"float64Array"]) {
		return [self floatingArrayFromBytes:data singles:[kind isEqualToString:@"float32Array"] key:key error:outError];
	}
	if ([kind isEqualToString:@"multiArray"]) {
		return [self multiArrayWrappingData:data entry:entry key:key error:outError];
	}
	if ([kind isEqualToString:@"pixelBuffer"]) {
		return [self pixelBufferWrappingData:data entry:entry key:key error:outError];
	}
	if ([kind isEqualToString:@"pcm"]) {
		return [self PCMBufferWrappingData:data entry:entry key:key error:outError];
	}
	if ([kind isEqualToString:@"cgImage"]) {
		return [self CGImageFromData:data key:key error:outError];
	}
	if ([kind isEqualToString:@"image"]) {
		CVPixelBufferRef pixelBuffer = [NFKImageCoding pixelBufferWithImageData:data];
		if (pixelBuffer == NULL) {
			return [self failForKey:key reason:@"the image could not be decoded" error:outError];
		}
		return (__bridge_transfer id)pixelBuffer;
	}
	return [self failForKey:key reason:[NSString stringWithFormat:@"the attachment kind \"%@\" is unknown", kind] error:outError];
}

/*! The shape's element count times elementSize when every extent is positive and the product stays
	within limit, or 0. */
static unsigned long long NFKWireShapeByteCount(NSArray *shape, NSInteger elementSize, unsigned long long limit)
{
	unsigned long long expected = elementSize > 0 && shape.count > 0 ? (unsigned long long)elementSize : 0;
	for (NSNumber *extent in shape) {
		BOOL valid = [extent isKindOfClass:NSNumber.class] && extent.longLongValue > 0 && expected <= limit;
		expected = valid ? expected * (unsigned long long)extent.longLongValue : 0;
	}
	return expected;
}

- (nullable id)multiArrayWrappingData:(NSData *)data entry:(NSDictionary *)entry key:(NSString *)key error:(NSError * _Nullable *)outError
{
	NSArray<NSNumber *> *shape = [entry[@"shape"] isKindOfClass:NSArray.class] ? entry[@"shape"] : nil;
	MLMultiArrayDataType dataType = (MLMultiArrayDataType)[self doubleIn:entry forKey:@"dataType"];
	NSInteger elementSize = (dataType & 0xFF) / 8;
	unsigned long long expected = NFKWireShapeByteCount(shape, elementSize, data.length);
	if (expected == 0 || expected != data.length) {
		return [self failForKey:key reason:@"the multi-array's shape does not match its bytes" error:outError];
	}
	NSMutableArray<NSNumber *> *strides = [NSMutableArray arrayWithCapacity:shape.count];
	NSInteger stride = 1;
	for (NSInteger axis = (NSInteger)shape.count - 1; axis >= 0; axis--) {
		[strides insertObject:@(stride) atIndex:0];
		stride *= shape[(NSUInteger)axis].integerValue;
	}
	NSError *createError = nil;
	MLMultiArray *array = [[MLMultiArray alloc] initWithDataPointer:(void *)data.bytes shape:shape dataType:dataType strides:strides
														deallocator:^(void *bytes) {
		(void)data;
	} error:&createError];
	return array ?: [self failForKey:key reason:createError.localizedDescription ?: @"a malformed multi-array" error:outError];
}

/*! A pixel buffer over an attachment's planes without a copy. Geometry the format rejects for wrapping
	falls back to a buffer the planes are copied into. */
- (nullable id)pixelBufferWrappingData:(NSData *)data entry:(NSDictionary *)entry key:(NSString *)key error:(NSError * _Nullable *)outError
{
	NSArray *planes = [entry[@"planes"] isKindOfClass:NSArray.class] ? entry[@"planes"] : nil;
	unsigned long long width = 0;
	unsigned long long height = 0;
	unsigned long long formatNumber = 0;
	BOOL whole = NFKWireWholeNumber(entry[@"width"], &width) && NFKWireWholeNumber(entry[@"height"], &height)
		&& NFKWireWholeNumber(entry[@"format"], &formatNumber) && formatNumber <= UINT32_MAX;
	for (id plane in planes) {
		unsigned long long field = 0;
		whole = whole && [plane isKindOfClass:NSDictionary.class] && NFKWireWholeNumber(plane[@"offset"], &field)
			&& NFKWireWholeNumber(plane[@"bytesPerRow"], &field) && NFKWireWholeNumber(plane[@"rows"], &field);
	}
	OSType format = (OSType)formatNumber;
	NSUInteger total = data.length;
	// Every plane must lie inside the attachment's length before a pointer into it reaches CoreVideo,
	// which neither checks a stride against the width nor knows the region's end.
	BOOL covered = whole && planes.count > 0 && width > 0 && height > 0
		&& NFKWirePlaneGeometryCovers(format, width, height, planes, ^double(NSDictionary *plane) {
			double offset = [plane[@"offset"] isKindOfClass:NSNumber.class] ? [plane[@"offset"] doubleValue] : -1;
			return offset >= 0 && offset <= (double)total ? (double)total - offset : 0;
		});
	if (!covered) {
		return [self failForKey:key reason:@"a malformed pixel buffer attachment" error:outError];
	}
	NSDictionary *description = CFBridgingRelease(CVPixelFormatDescriptionCreateWithPixelFormatType(kCFAllocatorDefault, format));
	NSArray<NSDictionary *> *layouts = description[(id)kCVPixelFormatPlanes];
	const uint8_t *bytes = data.bytes;
	CVPixelBufferRef buffer = NULL;
	void *retained = (void *)CFBridgingRetain(data);
	CVReturn status = kCVReturnError;
	if (layouts == nil) {
		NSDictionary *plane = planes.firstObject;
		status = CVPixelBufferCreateWithBytes(kCFAllocatorDefault, width, height, format, (void *)(bytes + [plane[@"offset"] unsignedLongValue]),
											  [plane[@"bytesPerRow"] unsignedLongValue], NFKWireReleaseAttachedBytes, retained, NULL, &buffer);
	} else {
		size_t count = planes.count;
		void *bases[count];
		size_t widths[count];
		size_t heights[count];
		size_t bytesPerRows[count];
		for (size_t index = 0; index < count; index++) {
			NSDictionary *plane = planes[index];
			double horizontal = MAX([layouts[index][(id)kCVPixelFormatHorizontalSubsampling] doubleValue], 1.0);
			double vertical = MAX([layouts[index][(id)kCVPixelFormatVerticalSubsampling] doubleValue], 1.0);
			bases[index] = (void *)(bytes + [plane[@"offset"] unsignedLongValue]);
			widths[index] = (size_t)ceil(width / horizontal);
			heights[index] = (size_t)ceil(height / vertical);
			bytesPerRows[index] = [plane[@"bytesPerRow"] unsignedLongValue];
		}
		status = CVPixelBufferCreateWithPlanarBytes(kCFAllocatorDefault, width, height, format, NULL, 0, count, bases, widths, heights,
													bytesPerRows, NFKWireReleaseAttachedPlanes, retained, NULL, &buffer);
	}
	if (status != kCVReturnSuccess || buffer == NULL) {
		CFRelease(retained);
		return [self failForKey:key reason:[NSString stringWithFormat:@"no pixel buffer of format %u wraps the attachment", (unsigned)format]
						  error:outError];
	}
	return (__bridge_transfer id)buffer;
}

/*! A PCM buffer over an attachment's samples without a copy. */
- (nullable id)PCMBufferWrappingData:(NSData *)data entry:(NSDictionary *)entry key:(NSString *)key error:(NSError * _Nullable *)outError
{
	AudioStreamBasicDescription description = {0};
	AVAudioFormat *format = [self PCMFormatFromEntry:entry description:&description];
	AVAudioFrameCount frames = (AVAudioFrameCount)[self doubleIn:entry forKey:@"frameLength"];
	NSArray *layouts = [entry[@"buffers"] isKindOfClass:NSArray.class] ? entry[@"buffers"] : nil;
	NSUInteger bytesPerFrame = description.mBytesPerFrame;
	UInt32 expectedBuffers = format == nil ? 0 : (format.isInterleaved ? 1 : format.channelCount);
	if (format == nil || frames == 0 || bytesPerFrame == 0 || layouts.count != expectedBuffers) {
		return [self failForKey:key reason:@"a malformed PCM attachment" error:outError];
	}
	AudioBufferList *list = calloc(1, offsetof(AudioBufferList, mBuffers) + sizeof(AudioBuffer) * layouts.count);
	list->mNumberBuffers = (UInt32)layouts.count;
	for (NSUInteger index = 0; index < layouts.count; index++) {
		NSDictionary *layout = [layouts[index] isKindOfClass:NSDictionary.class] ? layouts[index] : @{};
		unsigned long long offset = [layout[@"offset"] isKindOfClass:NSNumber.class] ? [layout[@"offset"] unsignedLongLongValue] : ULLONG_MAX;
		unsigned long long length = [layout[@"length"] isKindOfClass:NSNumber.class] ? [layout[@"length"] unsignedLongLongValue] : 0;
		// Every buffer holds exactly the frames named, inside the attachment.
		if (length != (unsigned long long)frames * bytesPerFrame || offset > data.length || length > data.length - offset) {
			free(list);
			return [self failForKey:key reason:@"a PCM attachment's samples do not match its frame count" error:outError];
		}
		list->mBuffers[index].mNumberChannels = format.isInterleaved ? format.channelCount : 1;
		list->mBuffers[index].mDataByteSize = (UInt32)length;
		list->mBuffers[index].mData = (uint8_t *)data.bytes + offset;
	}
	AVAudioPCMBuffer *buffer = [[AVAudioPCMBuffer alloc] initWithPCMFormat:format bufferListNoCopy:list deallocator:^(const AudioBufferList *bufferList) {
		free((void *)bufferList);
		(void)data;
	}];
	if (buffer == nil) {
		free(list);
		return [self failForKey:key reason:@"a malformed PCM attachment" error:outError];
	}
	buffer.frameLength = frames;
	return buffer;
}

- (nullable NSURL *)referencedFileURLInEntry:(NSDictionary *)entry key:(NSString *)key error:(NSError * _Nullable *)outError
{
	id<NFKWireAttachmentSource> source = self.attachmentSource;
	NSString *name = [entry[@"name"] isKindOfClass:NSString.class] ? entry[@"name"] : @"";
	NSError *sourceError = nil;
	NSURL *url = nil;
	if ([entry[@"index"] isKindOfClass:NSNumber.class] && [source respondsToSelector:@selector(fileURLForAttachmentAtIndex:name:error:)]) {
		url = [source fileURLForAttachmentAtIndex:[entry[@"index"] integerValue] name:name error:&sourceError];
	} else if ([entry[@"path"] isKindOfClass:NSString.class] && [entry[@"path"] isAbsolutePath]) {
		url = [source respondsToSelector:@selector(fileURLForPath:error:)] ? [source fileURLForPath:entry[@"path"] error:&sourceError]
																			: [NSURL fileURLWithPath:entry[@"path"]];
	} else {
		sourceError = nil;
		return [self failForKey:key reason:@"a file reference names no absolute path" error:outError];
	}
	return url ?: [self failForKey:key reason:sourceError.localizedDescription ?: @"the file reference is refused" error:outError];
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
	if ([tag isEqualToString:@"toolResult"]) {
		return [self toolResultFromEntry:entry key:key error:outError];
	}
	if ([tag isEqualToString:@"toolInvocation"]) {
		return [self toolInvocationFromEntry:entry key:key error:outError];
	}
	if ([tag isEqualToString:@"buffer"]) {
		return [self attachedValueFromEntry:entry key:key error:outError];
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

- (nullable NFKToolResult *)toolResultFromEntry:(NSDictionary *)entry key:(NSString *)key error:(NSError * _Nullable *)outError
{
	if (![entry[@"text"] isKindOfClass:NSString.class]) {
		return [self failForKey:key reason:@"a tool result without text" error:outError];
	}
	NSDictionary *outputs = [entry[@"outputs"] isKindOfClass:NSDictionary.class]
		? [self dictionaryFromJSONDictionary:entry[@"outputs"] error:outError] : @{};
	if (outputs == nil) {
		return nil;
	}
	NSMutableArray<NSData *> *images = [NSMutableArray array];
	for (NSString *encoded in [entry[@"images"] isKindOfClass:NSArray.class] ? entry[@"images"] : @[]) {
		NSData *png = [encoded isKindOfClass:NSString.class] ? [[NSData alloc] initWithBase64EncodedString:encoded options:0] : nil;
		if (png == nil) {
			return [self failForKey:key reason:@"a tool result image is not base64" error:outError];
		}
		[images addObject:png];
	}
	return [[NFKToolResult alloc] initWithText:entry[@"text"] outputs:outputs imagePNGs:images isError:[entry[@"isError"] boolValue]];
}

- (nullable NFKToolInvocation *)toolInvocationFromEntry:(NSDictionary *)entry key:(NSString *)key error:(NSError * _Nullable *)outError
{
	BOOL named = [entry[@"callIdentifier"] isKindOfClass:NSString.class] && [entry[@"toolName"] isKindOfClass:NSString.class];
	NSDictionary *wireResult = [entry[@"result"] isKindOfClass:NSDictionary.class] ? entry[@"result"] : nil;
	if (!named || wireResult == nil) {
		return [self failForKey:key reason:@"a malformed tool invocation" error:outError];
	}
	NFKToolResult *result = [self toolResultFromEntry:wireResult key:key error:outError];
	NSDictionary *arguments = [entry[@"arguments"] isKindOfClass:NSDictionary.class]
		? [self dictionaryFromJSONDictionary:entry[@"arguments"] error:outError] : @{};
	if (result == nil || arguments == nil) {
		return nil;
	}
	return [[NFKToolInvocation alloc] initWithCallIdentifier:entry[@"callIdentifier"] toolName:entry[@"toolName"] arguments:arguments
													  result:result durationSeconds:[entry[@"durationSeconds"] doubleValue]];
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
	} else if (self.attachmentSource != nil && (entry[@"path"] != nil || entry[@"index"] != nil)) {
		fileURL = [self referencedFileURLInEntry:entry key:key error:outError];
		if (fileURL == nil) {
			return nil;
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
	AVAudioFormat *format = [self PCMFormatFromEntry:entry description:&description];
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

/*! The format a PCM entry's stream description names, or nil when AVAudioFormat refuses it. The
	description is the entry's own, as read. */
- (nullable AVAudioFormat *)PCMFormatFromEntry:(NSDictionary *)entry description:(AudioStreamBasicDescription *)outDescription
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
	*outDescription = description;
	return [[AVAudioFormat alloc] initWithStreamDescription:&description channelLayout:layout];
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
