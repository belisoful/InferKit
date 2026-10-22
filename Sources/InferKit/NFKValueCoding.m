//
//  NFKValueCoding.m
//  InferKit
//

#import "NFKValueCoding.h"
#import "NFKInferenceWireCoding.h"
#import "NFKValueFiles.h"
#import "NFKWireValueObjects.h"
#import <InferKit/NFKModelVocabulary.h>
#import <InferKit/NFKAudioAsset.h>
#import <InferKit/NFKVideoAsset.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKMIDISequence.h>
#import <InferKit/NFKDecisionQuestion.h>
#import <InferKit/NFKErrors.h>
#import <AVFoundation/AVFoundation.h>

@interface NFKValueCoding ()
@property (nonatomic, strong) NFKInferenceWireCoding *coder;
@end

@implementation NFKValueCoding

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
		_coder.encodesValueObjectsAsJSON = YES;
	}
	return self;
}

- (instancetype)initWithDirectoryURL:(NSURL *)directoryURL
{
	self = [super init];
	if (self != nil) {
		_directoryURL = [directoryURL copy];
		_coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:directoryURL];
		_coder.encodesValueObjectsAsJSON = YES;
		_coder.fileDirectoryURL = directoryURL;
	}
	return self;
}

- (NSUInteger)inlineByteLimit
{
	return self.coder.inlineByteLimit;
}

- (void)setInlineByteLimit:(NSUInteger)inlineByteLimit
{
	self.coder.inlineByteLimit = inlineByteLimit;
}

- (NSArray<NSURL *> *)temporaryFileURLs
{
	return self.coder.temporaryFileURLs;
}

#pragma mark Requests, results, and values

- (nullable NSDictionary<NSString *, id> *)JSONObjectForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	return [self.coder JSONObjectForRequest:request error:error];
}

- (nullable NFKInferenceRequest *)requestFromJSONObject:(id)object error:(NSError **)error
{
	return [self.coder requestFromJSONObject:object error:error];
}

- (nullable NSDictionary<NSString *, id> *)JSONObjectForResult:(NFKInferenceResult *)result error:(NSError **)error
{
	return [self.coder JSONObjectForResult:result error:error];
}

- (nullable NFKInferenceResult *)resultFromJSONObject:(id)object error:(NSError **)error
{
	return [self.coder resultFromJSONObject:object error:error];
}

- (nullable id)JSONValueForValue:(id)value key:(NSString *)key error:(NSError **)error
{
	return [self.coder JSONValueForValue:value key:key error:error];
}

- (nullable id)valueFromJSONValue:(id)JSONValue key:(NSString *)key error:(NSError **)error
{
	return [self.coder valueFromJSONValue:JSONValue key:key error:error];
}

#pragma mark Port values

static id _Nullable NFKValueCodingFail(NSError **error, NSString *valueType, NSString *reason)
{
	if (error != NULL) {
		NSString *description = [NSString stringWithFormat:@"no %@ value can be read: %@", valueType, reason];
		*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceUnsupported
								 userInfo:@{ NSLocalizedDescriptionKey: description }];
	}
	return nil;
}

/*! The value-class tag whose arrays a value type holds, or nil. */
static NSString * _Nullable NFKValueCodingElementTag(NSString *valueType)
{
	NSDictionary<NSString *, NSString *> *tags = @{ NFKModelValueTypeDetections: @"detection",
													NFKModelValueTypeKeypoints: @"keypoint",
													NFKModelValueTypeClassifications: @"classification",
													NFKModelValueTypeSegments: @"audioSegment",
													NFKModelValueTypeWords: @"audioSegment",
													NFKModelValueTypeBeats: @"musicBeat" };
	return tags[valueType];
}

/*! Whether a value type holds an array of the single type it is the plural of. */
static NSString * _Nullable NFKValueCodingSingular(NSString *valueType)
{
	NSDictionary<NSString *, NSString *> *singulars = @{ NFKModelValueTypeImages: NFKModelValueTypeImage,
														 NFKModelValueTypeAudios: NFKModelValueTypeAudio,
														 NFKModelValueTypeVideos: NFKModelValueTypeVideo,
														 NFKModelValueTypeDocuments: NFKModelValueTypeDocument };
	return singulars[valueType];
}

- (nullable id)valueOfType:(NSString *)valueType fromFileURL:(NSURL *)url error:(NSError **)error
{
	if ([url.pathExtension.lowercaseString isEqualToString:@"json"]) {
		NSData *data = [NSData dataWithContentsOfURL:url options:0 error:error];
		id JSONValue = data != nil ? [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingFragmentsAllowed error:error] : nil;
		return JSONValue != nil ? [self valueOfType:valueType fromJSONValue:JSONValue error:error] : nil;
	}
	NSString *singular = NFKValueCodingSingular(valueType);
	if (singular != nil) {
		id value = [self valueOfType:singular fromFileURL:url error:error];
		return value != nil ? @[ value ] : nil;
	}
	if ([valueType isEqualToString:NFKModelValueTypeDocument]) {
		return [url checkResourceIsReachableAndReturnError:error] ? url : nil;
	}
	if ([valueType isEqualToString:NFKModelValueTypeVideo]) {
		return [url checkResourceIsReachableAndReturnError:error] ? [NFKVideoAsset videoAssetWithFileURL:url] : nil;
	}
	if ([valueType isEqualToString:NFKModelValueTypeAudio]) {
		return [self audioAssetForFileURL:url error:error];
	}
	if ([valueType isEqualToString:NFKModelValueTypeMIDI]) {
		return [NFKMIDISequence sequenceWithContentsOfURL:url error:error];
	}
	NSData *data = [NSData dataWithContentsOfURL:url options:NSDataReadingMappedIfSafe error:error];
	if (data == nil) {
		return nil;
	}
	if ([valueType isEqualToString:NFKModelValueTypeImage] || [valueType isEqualToString:NFKModelValueTypeMask]) {
		return [self pixelBufferFromImageData:data valueType:valueType error:error];
	}
	if ([valueType isEqualToString:NFKModelValueTypeEmbedding]) {
		return [self embeddingFromNPYData:data error:error];
	}
	NSString *text = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
	if (text == nil) {
		return NFKValueCodingFail(error, valueType, [NSString stringWithFormat:@"%@ is not UTF-8 text", url.lastPathComponent]);
	}
	if ([valueType isEqualToString:NFKModelValueTypeText]) {
		return text;
	}
	if ([valueType isEqualToString:NFKModelValueTypeTextList]) {
		NSMutableArray<NSString *> *lines = [NSMutableArray array];
		[text enumerateLinesUsingBlock:^(NSString *line, BOOL *stop) {
			if ([line stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet].length > 0) {
				[lines addObject:line];
			}
		}];
		return lines;
	}
	if ([valueType isEqualToString:NFKModelValueTypeNumber] || [valueType isEqualToString:NFKModelValueTypeInteger]
		|| [valueType isEqualToString:NFKModelValueTypeBoolean]) {
		return [self valueOfType:valueType fromJSONValue:[text stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet]
						   error:error];
	}
	return NFKValueCodingFail(error, valueType, @"the type is read from JSON only");
}

- (nullable NFKAudioAsset *)audioAssetForFileURL:(NSURL *)url error:(NSError **)error
{
	NSError *readError = nil;
	AVAudioFile *file = [[AVAudioFile alloc] initForReading:url error:&readError];
	if (file == nil) {
		return NFKValueCodingFail(error, NFKModelValueTypeAudio, [NSString stringWithFormat:@"%@ is not audio AVAudioFile reads: %@",
																  url.lastPathComponent, readError.localizedDescription]);
	}
	AVAudioFormat *format = file.fileFormat;
	return [NFKAudioAsset audioAssetWithFileURL:url
								durationSeconds:format.sampleRate > 0 ? (double)file.length / format.sampleRate : 0
									 sampleRate:format.sampleRate
								   channelCount:format.channelCount];
}

- (nullable id)pixelBufferFromImageData:(NSData *)data valueType:(NSString *)valueType error:(NSError **)error
{
	CVPixelBufferRef buffer = NFKValueFilePixelBufferFromPNG(data, kCVPixelFormatType_32BGRA, NULL);
	if (buffer == NULL) {
		buffer = [NFKImageCoding pixelBufferWithImageData:data];
	}
	return buffer != NULL ? (__bridge_transfer id)buffer : NFKValueCodingFail(error, valueType, @"the file is not an image ImageIO reads");
}

- (nullable NSArray<NSNumber *> *)embeddingFromNPYData:(NSData *)data error:(NSError **)error
{
	NSData *elements = nil;
	NSString *description = nil;
	NSArray<NSNumber *> *shape = nil;
	if (!NFKValueFileReadNPY(data, &elements, &description, &shape, error)) {
		return nil;
	}
	BOOL singles = [description isEqualToString:@"<f4"];
	if (shape.count != 1 || !(singles || [description isEqualToString:@"<f8"])) {
		return NFKValueCodingFail(error, NFKModelValueTypeEmbedding, @"the .npy file is not a vector of 32- or 64-bit floats");
	}
	NSMutableArray<NSNumber *> *vector = [NSMutableArray arrayWithCapacity:shape.firstObject.unsignedIntegerValue];
	for (NSUInteger index = 0; index < shape.firstObject.unsignedIntegerValue; index++) {
		[vector addObject:singles ? @(((const float *)elements.bytes)[index]) : @(((const double *)elements.bytes)[index])];
	}
	return vector;
}

- (nullable id)valueOfType:(NSString *)valueType fromJSONValue:(id)JSONValue error:(NSError **)error
{
	BOOL tagged = [JSONValue isKindOfClass:NSDictionary.class] && JSONValue[@"$nfk"] != nil;
	if (tagged) {
		return [self valueFromJSONValue:JSONValue key:valueType error:error];
	}
	NSString *singular = NFKValueCodingSingular(valueType);
	if (singular != nil) {
		NSArray *elements = [JSONValue isKindOfClass:NSArray.class] ? JSONValue : @[ JSONValue ];
		NSMutableArray *values = [NSMutableArray arrayWithCapacity:elements.count];
		for (id element in elements) {
			id value = [self valueOfType:singular fromJSONValue:element error:error];
			if (value == nil) {
				return nil;
			}
			[values addObject:value];
		}
		return values;
	}
	if ([valueType isEqualToString:NFKModelValueTypeText]) {
		return [JSONValue isKindOfClass:NSString.class] ? JSONValue : NFKValueCodingFail(error, valueType, @"the JSON is not a string");
	}
	if ([valueType isEqualToString:NFKModelValueTypeNumber] || [valueType isEqualToString:NFKModelValueTypeInteger]
		|| [valueType isEqualToString:NFKModelValueTypeBoolean]) {
		return [self scalarOfType:valueType fromJSONValue:JSONValue error:error];
	}
	if ([valueType isEqualToString:NFKModelValueTypeTextList]) {
		BOOL strings = [JSONValue isKindOfClass:NSArray.class];
		for (id element in strings ? JSONValue : @[]) {
			strings = strings && [element isKindOfClass:NSString.class];
		}
		return strings ? JSONValue : NFKValueCodingFail(error, valueType, @"the JSON is not an array of strings");
	}
	if ([valueType isEqualToString:NFKModelValueTypeEmbedding]) {
		BOOL numbers = [JSONValue isKindOfClass:NSArray.class];
		for (id element in numbers ? JSONValue : @[]) {
			numbers = numbers && [element isKindOfClass:NSNumber.class];
		}
		return numbers ? [self valueFromJSONValue:JSONValue key:valueType error:error]
					   : NFKValueCodingFail(error, valueType, @"the JSON is not an array of numbers");
	}
	NSString *elementTag = NFKValueCodingElementTag(valueType);
	if (elementTag != nil) {
		return [self valueObjectsTagged:elementTag fromJSONValue:JSONValue valueType:valueType error:error];
	}
	if ([valueType isEqualToString:NFKModelValueTypeMIDI]) {
		return [JSONValue isKindOfClass:NSDictionary.class] ? [self valueObjectTagged:@"midiSequence" entry:JSONValue valueType:valueType error:error]
															 : NFKValueCodingFail(error, valueType, @"the JSON is not an object");
	}
	if ([valueType isEqualToString:NFKModelValueTypeQuestions] || [valueType isEqualToString:NFKModelValueTypeAnswers]) {
		return [self decisionsOfType:valueType fromJSONValue:JSONValue error:error];
	}
	if ([valueType isEqualToString:NFKModelValueTypeImage] || [valueType isEqualToString:NFKModelValueTypeMask]) {
		return [self imageOfType:valueType fromJSONValue:JSONValue error:error];
	}
	if ([valueType isEqualToString:NFKModelValueTypeAudio] || [valueType isEqualToString:NFKModelValueTypeVideo]
		|| [valueType isEqualToString:NFKModelValueTypeDocument]) {
		NSURL *url = [self fileURLForPathValue:JSONValue valueType:valueType error:error];
		return url != nil ? [self valueOfType:valueType fromFileURL:url error:error] : nil;
	}
	return [self valueFromJSONValue:JSONValue key:valueType error:error];
}

- (nullable NSNumber *)scalarOfType:(NSString *)valueType fromJSONValue:(id)JSONValue error:(NSError **)error
{
	BOOL boolean = [valueType isEqualToString:NFKModelValueTypeBoolean];
	if ([JSONValue isKindOfClass:NSNumber.class]) {
		BOOL isBoolean = CFGetTypeID((__bridge CFTypeRef)JSONValue) == CFBooleanGetTypeID();
		if (boolean != isBoolean) {
			return NFKValueCodingFail(error, valueType, boolean ? @"the JSON is a number" : @"the JSON is a boolean");
		}
		BOOL whole = [JSONValue doubleValue] == floor([JSONValue doubleValue]);
		return [valueType isEqualToString:NFKModelValueTypeInteger] && !whole
			? NFKValueCodingFail(error, valueType, @"the number is not whole") : JSONValue;
	}
	NSString *text = [JSONValue isKindOfClass:NSString.class] ? [JSONValue lowercaseString] : nil;
	if (text == nil) {
		return NFKValueCodingFail(error, valueType, @"the JSON is neither a number nor text");
	}
	if (boolean) {
		NSDictionary<NSString *, NSNumber *> *words = @{ @"true": @YES, @"yes": @YES, @"1": @YES, @"false": @NO, @"no": @NO, @"0": @NO };
		return words[text] ?: NFKValueCodingFail(error, valueType, [NSString stringWithFormat:@"\"%@\" is not a boolean", JSONValue]);
	}
	NSScanner *scanner = [NSScanner scannerWithString:text];
	if ([valueType isEqualToString:NFKModelValueTypeInteger]) {
		long long whole = 0;
		return [scanner scanLongLong:&whole] && scanner.isAtEnd ? @(whole)
			: NFKValueCodingFail(error, valueType, [NSString stringWithFormat:@"\"%@\" is not a whole number", JSONValue]);
	}
	double number = 0;
	return [scanner scanDouble:&number] && scanner.isAtEnd ? @(number)
		: NFKValueCodingFail(error, valueType, [NSString stringWithFormat:@"\"%@\" is not a number", JSONValue]);
}

- (nullable id)valueObjectTagged:(NSString *)tag entry:(NSDictionary *)entry valueType:(NSString *)valueType error:(NSError **)error
{
	NSMutableDictionary *tagged = [entry mutableCopy];
	tagged[@"$nfk"] = tag;
	NSString *reason = nil;
	id value = NFKWireValueObjectFromJSON(tag, tagged, &reason);
	return value ?: NFKValueCodingFail(error, valueType, reason ?: @"an object is malformed");
}

- (nullable NSArray *)valueObjectsTagged:(NSString *)tag fromJSONValue:(id)JSONValue valueType:(NSString *)valueType error:(NSError **)error
{
	if (![JSONValue isKindOfClass:NSArray.class]) {
		return NFKValueCodingFail(error, valueType, @"the JSON is not an array");
	}
	NSMutableArray *values = [NSMutableArray arrayWithCapacity:[JSONValue count]];
	for (id element in JSONValue) {
		BOOL elementTagged = [element isKindOfClass:NSDictionary.class] && element[@"$nfk"] != nil;
		id value = nil;
		if (elementTagged) {
			value = [self valueFromJSONValue:element key:valueType error:error];
		} else if ([element isKindOfClass:NSDictionary.class]) {
			value = [self valueObjectTagged:tag entry:element valueType:valueType error:error];
		} else {
			value = NFKValueCodingFail(error, valueType, @"an element is not an object");
		}
		if (value == nil) {
			return nil;
		}
		[values addObject:value];
	}
	return values;
}

- (nullable NSDictionary *)decisionsOfType:(NSString *)valueType fromJSONValue:(id)JSONValue error:(NSError **)error
{
	if (![JSONValue isKindOfClass:NSDictionary.class]) {
		return NFKValueCodingFail(error, valueType, @"the JSON is not an object keyed by name");
	}
	BOOL questions = [valueType isEqualToString:NFKModelValueTypeQuestions];
	NSMutableDictionary *values = [NSMutableDictionary dictionary];
	for (NSString *name in JSONValue) {
		id entry = JSONValue[name];
		id value = nil;
		if ([entry isKindOfClass:NSDictionary.class] && entry[@"$nfk"] != nil) {
			value = [self valueFromJSONValue:entry key:valueType error:error];
		} else if ([entry isKindOfClass:NSDictionary.class]) {
			value = questions ? [NFKDecisionQuestion questionWithDictionary:entry]
							  : [self valueObjectTagged:@"decisionAnswer" entry:entry valueType:valueType error:error];
			if (value == nil && questions) {
				return NFKValueCodingFail(error, valueType, [NSString stringWithFormat:@"the question \"%@\" is malformed", name]);
			}
		} else {
			return NFKValueCodingFail(error, valueType, [NSString stringWithFormat:@"\"%@\" is not an object", name]);
		}
		if (value == nil) {
			return nil;
		}
		values[name] = value;
	}
	return values;
}

- (nullable id)imageOfType:(NSString *)valueType fromJSONValue:(id)JSONValue error:(NSError **)error
{
	NSString *text = [JSONValue isKindOfClass:NSString.class] ? JSONValue : nil;
	NSRange comma = text != nil ? [text rangeOfString:@","] : NSMakeRange(NSNotFound, 0);
	if (![text hasPrefix:@"data:"] || comma.location == NSNotFound || ![[text substringToIndex:comma.location] hasSuffix:@";base64"]) {
		return NFKValueCodingFail(error, valueType, @"the JSON is not a tagged image or a base64 data URL");
	}
	NSData *data = [[NSData alloc] initWithBase64EncodedString:[text substringFromIndex:comma.location + 1]
													   options:NSDataBase64DecodingIgnoreUnknownCharacters];
	return data != nil ? [self pixelBufferFromImageData:data valueType:valueType error:error]
					   : NFKValueCodingFail(error, valueType, @"the data URL is not base64");
}

- (nullable NSURL *)fileURLForPathValue:(id)JSONValue valueType:(NSString *)valueType error:(NSError **)error
{
	NSString *path = [JSONValue isKindOfClass:NSString.class] ? JSONValue : nil;
	if (path.length == 0) {
		return NFKValueCodingFail(error, valueType, @"the JSON names no file path");
	}
	if (self.directoryURL != nil) {
		NSError *resolveError = nil;
		NSURL *url = NFKValueFileResolve(self.directoryURL, path, &resolveError);
		return url ?: NFKValueCodingFail(error, valueType, resolveError.localizedDescription);
	}
	return path.isAbsolutePath ? [NSURL fileURLWithPath:path]
							   : NFKValueCodingFail(error, valueType, @"the inline form takes an absolute file path");
}

@end
