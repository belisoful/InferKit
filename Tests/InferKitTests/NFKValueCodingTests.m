//
//  NFKValueCodingTests.m
//  InferKitTests
//
//  The public value coding: the inline form, the file form's files and their round trips, the paths it
//  refuses, and the port values read from files and JSON.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <ImageIO/ImageIO.h>

@interface NFKValueCodingTests : XCTestCase
@property (nonatomic, strong) NSURL *directory;
@property (nonatomic, strong) NSURL *outside;
@end

@implementation NFKValueCodingTests

- (void)setUp
{
	[super setUp];
	NSURL *root = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:NSUUID.UUID.UUIDString isDirectory:YES];
	self.directory = [root URLByAppendingPathComponent:@"values" isDirectory:YES];
	self.outside = [root URLByAppendingPathComponent:@"elsewhere" isDirectory:YES];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:self.directory withIntermediateDirectories:YES attributes:nil error:NULL]);
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:self.outside withIntermediateDirectories:YES attributes:nil error:NULL]);
}

- (void)tearDown
{
	[NSFileManager.defaultManager removeItemAtURL:self.directory.URLByDeletingLastPathComponent error:NULL];
	[super tearDown];
}

#pragma mark Helpers

/*! A pixel buffer of format whose every byte, alpha included, follows a pattern of its position. */
- (CVPixelBufferRef)patternedBufferOfFormat:(OSType)format width:(size_t)width height:(size_t)height CF_RETURNS_RETAINED
{
	CVPixelBufferRef buffer = NULL;
	XCTAssertEqual(CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, NULL, &buffer), kCVReturnSuccess);
	CVPixelBufferLockBaseAddress(buffer, 0);
	uint8_t *base = CVPixelBufferGetBaseAddress(buffer);
	size_t bytesPerRow = CVPixelBufferGetBytesPerRow(buffer);
	for (size_t row = 0; row < height; row++) {
		for (size_t column = 0; column < bytesPerRow; column++) {
			base[row * bytesPerRow + column] = (uint8_t)((row * 37 + column * 11 + 5) % 256);
		}
	}
	CVPixelBufferUnlockBaseAddress(buffer, 0);
	return buffer;
}

/*! Whether two pixel buffers hold the same format, size, and bytes in each row's pixels. */
- (BOOL)buffer:(CVPixelBufferRef)left equals:(CVPixelBufferRef)right
{
	OSType format = CVPixelBufferGetPixelFormatType(left);
	size_t width = CVPixelBufferGetWidth(left);
	size_t height = CVPixelBufferGetHeight(left);
	if (format != CVPixelBufferGetPixelFormatType(right) || width != CVPixelBufferGetWidth(right) || height != CVPixelBufferGetHeight(right)) {
		return NO;
	}
	NSDictionary *description = CFBridgingRelease(CVPixelFormatDescriptionCreateWithPixelFormatType(kCFAllocatorDefault, format));
	size_t rowBytes = width * [description[(id)kCVPixelFormatBitsPerBlock] unsignedLongValue] / 8;
	CVPixelBufferLockBaseAddress(left, kCVPixelBufferLock_ReadOnly);
	CVPixelBufferLockBaseAddress(right, kCVPixelBufferLock_ReadOnly);
	BOOL equal = YES;
	for (size_t row = 0; row < height && equal; row++) {
		const uint8_t *a = (const uint8_t *)CVPixelBufferGetBaseAddress(left) + row * CVPixelBufferGetBytesPerRow(left);
		const uint8_t *b = (const uint8_t *)CVPixelBufferGetBaseAddress(right) + row * CVPixelBufferGetBytesPerRow(right);
		equal = memcmp(a, b, rowBytes) == 0;
	}
	CVPixelBufferUnlockBaseAddress(left, kCVPixelBufferLock_ReadOnly);
	CVPixelBufferUnlockBaseAddress(right, kCVPixelBufferLock_ReadOnly);
	return equal;
}

- (AVAudioPCMBuffer *)toneOfFormat:(AVAudioCommonFormat)commonFormat channels:(AVAudioChannelCount)channels interleaved:(BOOL)interleaved
{
	return [self tone:[[AVAudioFormat alloc] initWithCommonFormat:commonFormat sampleRate:24000 channels:channels interleaved:interleaved]];
}

- (AVAudioPCMBuffer *)tone:(AVAudioFormat *)format
{
	AVAudioPCMBuffer *buffer = [[AVAudioPCMBuffer alloc] initWithPCMFormat:format frameCapacity:1000];
	buffer.frameLength = 1000;
	const AudioBufferList *list = buffer.audioBufferList;
	for (UInt32 index = 0; index < list->mNumberBuffers; index++) {
		uint8_t *bytes = list->mBuffers[index].mData;
		for (UInt32 offset = 0; offset < list->mBuffers[index].mDataByteSize; offset++) {
			bytes[offset] = (uint8_t)((offset * 13 + index * 101) % 251);
		}
	}
	return buffer;
}

- (BOOL)PCMBuffer:(AVAudioPCMBuffer *)left equals:(AVAudioPCMBuffer *)right
{
	if (![left.format isEqual:right.format] || left.frameLength != right.frameLength) {
		return NO;
	}
	NSUInteger bytes = left.frameLength * left.format.streamDescription->mBytesPerFrame;
	for (UInt32 index = 0; index < left.audioBufferList->mNumberBuffers; index++) {
		if (memcmp(left.audioBufferList->mBuffers[index].mData, right.audioBufferList->mBuffers[index].mData, bytes) != 0) {
			return NO;
		}
	}
	return YES;
}

- (nullable id)throughJSON:(id)object
{
	NSData *data = [NSJSONSerialization dataWithJSONObject:object options:0 error:NULL];
	return data != nil ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
}

- (NSURL *)write:(NSData *)data named:(NSString *)name
{
	NSURL *url = [self.outside URLByAppendingPathComponent:name];
	XCTAssertTrue([data writeToURL:url atomically:YES]);
	return url;
}

- (void)assertRefusal:(nullable NSError *)error
{
	XCTAssertEqualObjects(error.domain, NFKInferenceErrorDomain, @"%@", error);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported, @"%@", error);
}

#pragma mark The inline form

- (void)testTheInlineFormCarriesAResultAndItsValueClassesAsJSON
{
	CVPixelBufferRef image = [self patternedBufferOfFormat:kCVPixelFormatType_32BGRA width:5 height:3];
	NFKDetection *cat = [[NFKDetection alloc] initWithLabel:@"cat" classIndex:3 confidence:0.875
												boundingBox:CGRectMake(0.125, 0.25, 0.5, 0.5) quadrilateral:nil];
	NFKMIDISequence *melody = [NFKMIDISequence sequenceWithNotes:@[ [NFKMIDINote noteWithPitch:60 startSeconds:0 endSeconds:0.5 velocity:90] ]];
	NSArray<NSNumber *> *embedding = @[ @(0.25f), @(-1.5f), @(3.0f) ];
	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"hello", NFKOutputImage: (__bridge id)image,
																		  NFKOutputDetections: @[ cat ], NFKOutputMIDI: melody,
																		  NFKOutputEmbedding: embedding }];
	NFKValueCoding *coding = [[NFKValueCoding alloc] init];
	NSError *error = nil;
	NSDictionary *encoded = [self throughJSON:[coding JSONObjectForResult:result error:&error]];
	XCTAssertNotNil(encoded, @"%@", error);
	XCTAssertEqualObjects(encoded[@"outputs"][NFKOutputDetections][0][@"$nfk"], @"detection", @"value classes travel as JSON objects");
	NFKInferenceResult *decoded = [coding resultFromJSONObject:encoded error:&error];
	XCTAssertNotNil(decoded, @"%@", error);
	XCTAssertEqualObjects(decoded.outputs[NFKOutputText], @"hello");
	XCTAssertEqualObjects(decoded.outputs[NFKOutputDetections], @[ cat ]);
	XCTAssertEqualObjects(decoded.outputs[NFKOutputMIDI], melody);
	XCTAssertEqualObjects(decoded.outputs[NFKOutputEmbedding], embedding);
	XCTAssertTrue([self buffer:image equals:(__bridge CVPixelBufferRef)decoded.outputs[NFKOutputImage]]);
	XCTAssertNil(coding.directoryURL);
	CVPixelBufferRelease(image);
}

#pragma mark The file form

- (void)testTheFileFormWritesEightBitImagesAsPNGsThatKeepEveryByte
{
	NFKValueCoding *coding = [[NFKValueCoding alloc] initWithDirectoryURL:self.directory];
	OSType formats[] = { kCVPixelFormatType_32BGRA, kCVPixelFormatType_32ARGB, kCVPixelFormatType_OneComponent8 };
	NSMutableSet<NSString *> *paths = [NSMutableSet set];
	for (size_t index = 0; index < sizeof(formats) / sizeof(formats[0]); index++) {
		CVPixelBufferRef image = [self patternedBufferOfFormat:formats[index] width:17 height:9];
		NSError *error = nil;
		NSDictionary *entry = [self throughJSON:[coding JSONValueForValue:(__bridge id)image key:NFKOutputImage error:&error]];
		XCTAssertEqualObjects(entry[@"$nfk"], @"pixelBuffer", @"%@", error);
		NSString *path = entry[@"path"];
		XCTAssertEqualObjects(path.pathExtension, @"png");
		XCTAssertNil(entry[@"planes"], @"a PNG carries the pixels");
		[paths addObject:path];
		NSURL *file = [self.directory URLByAppendingPathComponent:path];
		CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)file, NULL);
		XCTAssertEqualObjects((__bridge NSString *)CGImageSourceGetType(source), @"public.png");
		CFRelease(source);
		id decoded = [coding valueFromJSONValue:entry key:NFKOutputImage error:&error];
		XCTAssertNotNil(decoded, @"%@", error);
		XCTAssertTrue([self buffer:image equals:(__bridge CVPixelBufferRef)decoded], @"format %u comes back byte for byte", (unsigned)formats[index]);
		CVPixelBufferRelease(image);
	}
	XCTAssertEqualObjects(paths, ([NSSet setWithArray:@[ @"image.png", @"image-2.png", @"image-3.png" ]]),
						  @"each file is named after its key");
}

- (void)testTheFileFormWritesOtherPixelFormatsAsTheirPlanes
{
	NFKValueCoding *coding = [[NFKValueCoding alloc] initWithDirectoryURL:self.directory];
	CVPixelBufferRef depth = [self patternedBufferOfFormat:kCVPixelFormatType_OneComponent32Float width:6 height:4];
	NSError *error = nil;
	NSDictionary *entry = [self throughJSON:[coding JSONValueForValue:(__bridge id)depth key:@"depth" error:&error]];
	XCTAssertEqualObjects([entry[@"path"] pathExtension], @"bin", @"%@", error);
	XCTAssertEqual([entry[@"planes"] count], 1u);
	id decoded = [coding valueFromJSONValue:entry key:@"depth" error:&error];
	XCTAssertTrue([self buffer:depth equals:(__bridge CVPixelBufferRef)decoded], @"%@", error);
	CVPixelBufferRelease(depth);
}

- (void)testTheFileFormWritesPCMAsAWAVThatAVAudioFileOpens
{
	NFKValueCoding *coding = [[NFKValueCoding alloc] initWithDirectoryURL:self.directory];
	AVAudioChannelLayout *three = [AVAudioChannelLayout layoutWithLayoutTag:kAudioChannelLayoutTag_DiscreteInOrder | 3];
	AVAudioFormat *surround = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32 sampleRate:24000 interleaved:NO
															  channelLayout:three];
	NSArray<AVAudioPCMBuffer *> *buffers = @[ [self toneOfFormat:AVAudioPCMFormatFloat32 channels:2 interleaved:NO],
											  [self toneOfFormat:AVAudioPCMFormatInt16 channels:1 interleaved:YES],
											  [self toneOfFormat:AVAudioPCMFormatInt32 channels:2 interleaved:YES],
											  [self tone:surround] ];
	for (AVAudioPCMBuffer *buffer in buffers) {
		NSError *error = nil;
		NSDictionary *entry = [self throughJSON:[coding JSONValueForValue:buffer key:NFKOutputAudio error:&error]];
		XCTAssertEqualObjects([entry[@"path"] pathExtension], @"wav", @"%@", error);
		AVAudioFile *file = [[AVAudioFile alloc] initForReading:[self.directory URLByAppendingPathComponent:entry[@"path"]] error:&error];
		XCTAssertNotNil(file, @"%@", error);
		XCTAssertEqual(file.length, 1000);
		XCTAssertEqual(file.fileFormat.channelCount, buffer.format.channelCount);
		XCTAssertEqual(file.fileFormat.sampleRate, 24000);
		AVAudioPCMBuffer *decoded = [coding valueFromJSONValue:entry key:NFKOutputAudio error:&error];
		XCTAssertTrue([self PCMBuffer:buffer equals:decoded], @"%@ comes back sample for sample: %@", buffer.format, error);
	}
}

- (void)testTheFileFormCopiesAFileFromOutsideAndReferencesOneInside
{
	NFKValueCoding *coding = [[NFKValueCoding alloc] initWithDirectoryURL:self.directory];
	NSData *bytes = [@"RIFF-not-really" dataUsingEncoding:NSUTF8StringEncoding];
	NSURL *recording = [self write:bytes named:@"take one.wav"];
	NFKAudioAsset *asset = [NFKAudioAsset audioAssetWithFileURL:recording durationSeconds:2.5 sampleRate:16000 channelCount:1];
	NSError *error = nil;
	NSDictionary *entry = [self throughJSON:[coding JSONValueForValue:asset key:NFKOutputAudio error:&error]];
	XCTAssertEqualObjects(entry[@"path"], @"take_one.wav", @"%@", error);
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:[self.directory URLByAppendingPathComponent:@"take_one.wav"]], bytes);
	NFKAudioAsset *decoded = [coding valueFromJSONValue:entry key:NFKOutputAudio error:&error];
	XCTAssertEqualObjects(decoded.fileURL.URLByResolvingSymlinksInPath, [self.directory URLByAppendingPathComponent:@"take_one.wav"].URLByResolvingSymlinksInPath);
	XCTAssertEqual(decoded.durationSeconds, 2.5);
	XCTAssertEqual(decoded.sampleRate, 16000);

	NSURL *inside = [self.directory URLByAppendingPathComponent:@"notes.txt"];
	XCTAssertTrue([bytes writeToURL:inside atomically:YES]);
	NSDictionary *reference = [coding JSONValueForValue:inside key:NFKInputDocument error:&error];
	XCTAssertEqualObjects(reference[@"path"], @"notes.txt", @"a file inside the directory is named, not copied");
	NSArray *contents = [NSFileManager.defaultManager contentsOfDirectoryAtPath:self.directory.path error:NULL];
	XCTAssertEqual(contents.count, 2u);
}

- (void)testLargeArraysAndDataGoToNumPyAndBinaryFiles
{
	NFKValueCoding *coding = [[NFKValueCoding alloc] initWithDirectoryURL:self.directory];
	coding.inlineByteLimit = 16;
	NSArray<NSNumber *> *embedding = @[ @(0.5f), @(1.5f), @(2.5f), @(3.5f), @(4.5f), @(5.5f) ];
	NSArray<NSNumber *> *small = @[ @(0.5f), @(1.5f) ];
	NSData *blob = [NSMutableData dataWithLength:100];
	NSError *error = nil;
	MLMultiArray *array = [[MLMultiArray alloc] initWithShape:@[ @2, @3 ] dataType:MLMultiArrayDataTypeFloat32 error:&error];
	for (NSInteger index = 0; index < 6; index++) {
		array[index] = @(index * 0.25);
	}
	NSDictionary *encoded = [self throughJSON:[coding JSONObjectForResult:[NFKInferenceResult resultWithOutputs:@{
		NFKOutputEmbedding: embedding, @"small": small, @"blob": blob, @"tensor": array }] error:&error]];
	NSDictionary *outputs = encoded[@"outputs"];
	XCTAssertEqualObjects(outputs[NFKOutputEmbedding][@"path"], @"embedding.npy", @"%@", error);
	XCTAssertEqualObjects(outputs[@"tensor"][@"path"], @"tensor.npy");
	XCTAssertEqualObjects(outputs[@"blob"][@"path"], @"blob.bin");
	XCTAssertNil(outputs[@"small"][@"path"], @"an array within the limit stays in the JSON");
	NSData *header = [[NSData dataWithContentsOfURL:[self.directory URLByAppendingPathComponent:@"embedding.npy"]] subdataWithRange:NSMakeRange(0, 6)];
	XCTAssertEqualObjects(header, [NSData dataWithBytes:"\x93NUMPY" length:6]);

	NFKInferenceResult *decoded = [coding resultFromJSONObject:encoded error:&error];
	XCTAssertEqualObjects(decoded.outputs[NFKOutputEmbedding], embedding, @"%@", error);
	XCTAssertEqualObjects(decoded.outputs[@"small"], small);
	XCTAssertEqualObjects(decoded.outputs[@"blob"], blob);
	MLMultiArray *tensor = decoded.outputs[@"tensor"];
	XCTAssertEqualObjects(tensor.shape, (@[ @2, @3 ]));
	XCTAssertEqualObjects(tensor[5], @(1.25f));
}

- (void)testTheFileFormWritesAMIDISequenceBesideItsFields
{
	NFKValueCoding *coding = [[NFKValueCoding alloc] initWithDirectoryURL:self.directory];
	NFKMIDISequence *melody = [NFKMIDISequence sequenceWithNotes:@[ [NFKMIDINote noteWithPitch:60 startSeconds:0 endSeconds:0.5 velocity:90],
																	 [NFKMIDINote noteWithPitch:64 startSeconds:0.5 endSeconds:1 velocity:80] ]];
	NSError *error = nil;
	NSDictionary *entry = [self throughJSON:[coding JSONValueForValue:melody key:NFKOutputMIDI error:&error]];
	XCTAssertEqualObjects(entry[@"$nfk"], @"midiSequence", @"%@", error);
	XCTAssertEqual([entry[@"notes"] count], 2u);
	XCTAssertEqualObjects(entry[@"path"], @"midi.mid");
	NFKMIDISequence *file = [NFKMIDISequence sequenceWithContentsOfURL:[self.directory URLByAppendingPathComponent:@"midi.mid"] error:&error];
	XCTAssertEqual(file.notes.count, 2u, @"%@", error);
	XCTAssertEqualObjects([coding valueFromJSONValue:entry key:NFKOutputMIDI error:&error], melody, @"the fields decode exactly");
}

- (void)testTheFileFormRefusesAPathThatLeavesItsDirectory
{
	NFKValueCoding *coding = [[NFKValueCoding alloc] initWithDirectoryURL:self.directory];
	NSURL *secret = [self write:[@"secret" dataUsingEncoding:NSUTF8StringEncoding] named:@"secret.bin"];
	NSURL *link = [self.directory URLByAppendingPathComponent:@"link.bin"];
	XCTAssertTrue([NSFileManager.defaultManager createSymbolicLinkAtURL:link withDestinationURL:secret error:NULL]);
	for (NSString *path in @[ @"../elsewhere/secret.bin", secret.path, @"link.bin", @"absent.bin" ]) {
		NSError *error = nil;
		NSDictionary *entry = @{ @"$nfk": @"data", @"path": path };
		XCTAssertNil([coding valueFromJSONValue:entry key:@"blob" error:&error], @"%@", path);
		[self assertRefusal:error];
	}
}

- (void)testARequestRoundTripsThroughTheFileForm
{
	NFKValueCoding *coding = [[NFKValueCoding alloc] initWithDirectoryURL:self.directory];
	CVPixelBufferRef image = [self patternedBufferOfFormat:kCVPixelFormatType_32BGRA width:8 height:8];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a cat", NFKInputImage: (__bridge id)image }
															   parameters:@{ @"steps": @4 } outputModality:NFKModalityImage];
	NSError *error = nil;
	NSDictionary *encoded = [self throughJSON:[coding JSONObjectForRequest:request error:&error]];
	NFKInferenceRequest *decoded = [coding requestFromJSONObject:encoded error:&error];
	XCTAssertNotNil(decoded, @"%@", error);
	XCTAssertEqualObjects([decoded inputForKey:NFKInputPrompt], @"a cat");
	XCTAssertEqualObjects([decoded parameterForKey:@"steps"], @4);
	XCTAssertEqual(decoded.outputModality, NFKModalityImage);
	XCTAssertTrue([self buffer:image equals:(__bridge CVPixelBufferRef)[decoded inputForKey:NFKInputImage]]);
	CVPixelBufferRelease(image);
}

#pragma mark Port values

- (void)testAFileBecomesThePortValueItsTypeTakes
{
	NFKValueCoding *coding = [[NFKValueCoding alloc] init];
	NSError *error = nil;
	CVPixelBufferRef picture = [self patternedBufferOfFormat:kCVPixelFormatType_32BGRA width:7 height:5];
	NSURL *png = [self write:[NFKImageCoding PNGDataForImage:(__bridge id)picture] named:@"picture.png"];
	CVPixelBufferRef image = (__bridge CVPixelBufferRef)[coding valueOfType:NFKModelValueTypeImage fromFileURL:png error:&error];
	XCTAssertEqual(CVPixelBufferGetPixelFormatType(image), kCVPixelFormatType_32BGRA, @"%@", error);
	XCTAssertEqual(CVPixelBufferGetWidth(image), 7u);
	NSArray *images = [coding valueOfType:NFKModelValueTypeImages fromFileURL:png error:&error];
	XCTAssertEqual(images.count, 1u);
	CVPixelBufferRelease(picture);

	AVAudioPCMBuffer *tone = [self toneOfFormat:AVAudioPCMFormatFloat32 channels:2 interleaved:NO];
	NSURL *wave = [self.outside URLByAppendingPathComponent:@"tone.caf"];
	AVAudioFile *writer = [[AVAudioFile alloc] initForWriting:wave settings:tone.format.settings error:&error];
	XCTAssertTrue([writer writeFromBuffer:tone error:&error], @"%@", error);
	writer = nil;
	NFKAudioAsset *audio = [coding valueOfType:NFKModelValueTypeAudio fromFileURL:wave error:&error];
	XCTAssertEqual(audio.sampleRate, 24000, @"%@", error);
	XCTAssertEqual(audio.channelCount, 2);
	XCTAssertEqualWithAccuracy(audio.durationSeconds, 1000.0 / 24000.0, 1e-9);

	NFKMIDISequence *melody = [NFKMIDISequence sequenceWithNotes:@[ [NFKMIDINote noteWithPitch:67 startSeconds:0 endSeconds:0.5 velocity:90] ]];
	NSURL *midi = [self write:[melody standardMIDIFileData] named:@"melody.mid"];
	XCTAssertEqual([[coding valueOfType:NFKModelValueTypeMIDI fromFileURL:midi error:&error] notes].firstObject.pitch, 67, @"%@", error);

	NSURL *text = [self write:[@"first\n\nsecond\n" dataUsingEncoding:NSUTF8StringEncoding] named:@"lines.txt"];
	XCTAssertEqualObjects([coding valueOfType:NFKModelValueTypeText fromFileURL:text error:&error], @"first\n\nsecond\n");
	XCTAssertEqualObjects([coding valueOfType:NFKModelValueTypeTextList fromFileURL:text error:&error], (@[ @"first", @"second" ]));
	NSURL *count = [self write:[@" 42\n" dataUsingEncoding:NSUTF8StringEncoding] named:@"count.txt"];
	XCTAssertEqualObjects([coding valueOfType:NFKModelValueTypeInteger fromFileURL:count error:&error], @42, @"%@", error);
	XCTAssertEqualObjects([coding valueOfType:NFKModelValueTypeDocument fromFileURL:text error:&error], text);

	NSURL *detections = [self write:[@"[{\"label\": \"cat\", \"classIndex\": 3, \"confidence\": 0.5, "
									  "\"boundingBox\": {\"x\": 0.25, \"y\": 0.25, \"width\": 0.5, \"height\": 0.5}}]" dataUsingEncoding:NSUTF8StringEncoding]
							  named:@"detections.json"];
	NFKDetection *expected = [[NFKDetection alloc] initWithLabel:@"cat" classIndex:3 confidence:0.5
													 boundingBox:CGRectMake(0.25, 0.25, 0.5, 0.5) quadrilateral:nil];
	XCTAssertEqualObjects([coding valueOfType:NFKModelValueTypeDetections fromFileURL:detections error:&error], @[ expected ], @"%@", error);

	NSURL *notImage = [self write:[@"plain text" dataUsingEncoding:NSUTF8StringEncoding] named:@"plain.png"];
	XCTAssertNil([coding valueOfType:NFKModelValueTypeImage fromFileURL:notImage error:&error]);
	[self assertRefusal:error];
	error = nil;
	XCTAssertNil([coding valueOfType:NFKModelValueTypeStructured fromFileURL:text error:&error], @"structured values come from JSON only");
	[self assertRefusal:error];
}

- (void)testAVectorNumPyFileBecomesAnEmbedding
{
	NFKValueCoding *files = [[NFKValueCoding alloc] initWithDirectoryURL:self.directory];
	files.inlineByteLimit = 0;
	NSArray<NSNumber *> *vector = @[ @(0.125f), @(-2.0f), @(7.5f) ];
	NSError *error = nil;
	NSDictionary *entry = [files JSONValueForValue:vector key:NFKOutputEmbedding error:&error];
	NSURL *npy = [self.directory URLByAppendingPathComponent:entry[@"path"]];
	XCTAssertEqualObjects([[[NFKValueCoding alloc] init] valueOfType:NFKModelValueTypeEmbedding fromFileURL:npy error:&error], vector, @"%@", error);
}

- (void)testJSONBecomesThePortValueItsTypeTakes
{
	NFKValueCoding *coding = [[NFKValueCoding alloc] initWithDirectoryURL:self.directory];
	NSError *error = nil;
	XCTAssertEqualObjects([coding valueOfType:NFKModelValueTypeNumber fromJSONValue:@"2.5" error:&error], @2.5);
	XCTAssertEqualObjects([coding valueOfType:NFKModelValueTypeBoolean fromJSONValue:@"Yes" error:&error], @YES);
	XCTAssertEqualObjects([coding valueOfType:NFKModelValueTypeBoolean fromJSONValue:@NO error:&error], @NO);
	XCTAssertNil([coding valueOfType:NFKModelValueTypeInteger fromJSONValue:@2.5 error:&error]);
	[self assertRefusal:error];
	XCTAssertNil([coding valueOfType:NFKModelValueTypeBoolean fromJSONValue:@1 error:&error], @"a number is not a boolean");
	XCTAssertNil([coding valueOfType:NFKModelValueTypeTextList fromJSONValue:(@[ @"a", @1 ]) error:&error]);

	NSDictionary *questions = @{ @"tone": @{ @"type": @"choice", @"instructions": @"How does it sound?",
											 @"criteria": @{ @"calm": NSNull.null, @"tense": NSNull.null } } };
	NSDictionary *read = [coding valueOfType:NFKModelValueTypeQuestions fromJSONValue:questions error:&error];
	XCTAssertTrue([read[@"tone"] isKindOfClass:NFKDecisionQuestion.class], @"%@", error);

	CVPixelBufferRef picture = [self patternedBufferOfFormat:kCVPixelFormatType_32BGRA width:3 height:2];
	NSString *dataURL = [NFKImageCoding dataURLForImage:(__bridge id)picture];
	CVPixelBufferRef image = (__bridge CVPixelBufferRef)[coding valueOfType:NFKModelValueTypeImage fromJSONValue:dataURL error:&error];
	XCTAssertEqual(CVPixelBufferGetHeight(image), 2u, @"%@", error);
	CVPixelBufferRelease(picture);

	NSURL *inside = [self.directory URLByAppendingPathComponent:@"report.pdf"];
	XCTAssertTrue([[NSData data] writeToURL:inside atomically:YES]);
	NSArray *documents = [coding valueOfType:NFKModelValueTypeDocuments fromJSONValue:@[ @"report.pdf" ] error:&error];
	XCTAssertEqualObjects([documents.firstObject URLByResolvingSymlinksInPath], inside.URLByResolvingSymlinksInPath, @"%@", error);
	XCTAssertNil([coding valueOfType:NFKModelValueTypeDocument fromJSONValue:@"../elsewhere/x.pdf" error:&error]);
	[self assertRefusal:error];
}

#pragma mark Malformed input

/*! An .npy file whose header is header, padded as the format pads it, with payload after it. */
- (NSData *)npyWithVersion:(uint8_t)version header:(NSString *)header payload:(NSData *)payload
{
	NSMutableString *padded = [header mutableCopy];
	while ((10 + padded.length + 1) % 64 != 0) {
		[padded appendString:@" "];
	}
	[padded appendString:@"\n"];
	NSMutableData *data = [NSMutableData dataWithBytes:"\x93NUMPY" length:6];
	uint8_t versionBytes[2] = { version, 0 };
	[data appendBytes:versionBytes length:2];
	uint16_t length = (uint16_t)padded.length;
	[data appendBytes:&length length:2];
	[data appendData:[padded dataUsingEncoding:NSASCIIStringEncoding]];
	[data appendData:payload];
	return data;
}

- (void)testMalformedNumPyFilesAreRefusedWithoutAnException
{
	NFKValueCoding *coding = [[NFKValueCoding alloc] init];
	NSData *four = [NSMutableData dataWithLength:4];
	NSArray<NSData *> *files = @[
		[self npyWithVersion:1 header:@"{'descr': '', 'fortran_order': False, 'shape': (1,), }" payload:four],
		[self npyWithVersion:1 header:@"{'descr': 'x', 'fortran_order': False, 'shape': (1,), }" payload:four],
		[self npyWithVersion:1 header:@"{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }" payload:four],
		[self npyWithVersion:9 header:@"{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }" payload:four],
		[NSData dataWithBytes:"\x93NUMPY\x01" length:7],
	];
	for (NSData *file in files) {
		NSError *error = nil;
		XCTAssertNil([coding valueOfType:NFKModelValueTypeEmbedding fromFileURL:[self write:file named:@"vector.npy"] error:&error]);
		[self assertRefusal:error];
	}
	NSError *error = nil;
	NSURL *good = [self write:[self npyWithVersion:1 header:@"{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }" payload:four]
						named:@"good.npy"];
	XCTAssertEqualObjects([coding valueOfType:NFKModelValueTypeEmbedding fromFileURL:good error:&error], @[ @0 ], @"%@", error);
}

- (void)testATruncatedWAVIsRefused
{
	NFKValueCoding *coding = [[NFKValueCoding alloc] initWithDirectoryURL:self.directory];
	NSError *error = nil;
	NSDictionary *entry = [self throughJSON:[coding JSONValueForValue:[self toneOfFormat:AVAudioPCMFormatInt16 channels:1 interleaved:YES]
																  key:NFKOutputAudio error:&error]];
	NSURL *wave = [self.directory URLByAppendingPathComponent:entry[@"path"]];
	NSData *whole = [NSData dataWithContentsOfURL:wave];
	for (NSNumber *length in @[ @8, @30, @(whole.length - 100) ]) {
		XCTAssertTrue([[whole subdataWithRange:NSMakeRange(0, length.unsignedIntegerValue)] writeToURL:wave atomically:NO]);
		error = nil;
		XCTAssertNil([coding valueFromJSONValue:entry key:NFKOutputAudio error:&error], @"%@ bytes", length);
		[self assertRefusal:error];
	}
}

- (void)testAnImageOfTooManyPixelsIsRefusedBeforeItIsDecoded
{
	// One pixel past 8192 × 8192 on each side. The gray zeros compress to a few kilobytes.
	CGColorSpaceRef gray = CGColorSpaceCreateDeviceGray();
	CGContextRef context = CGBitmapContextCreate(NULL, 8193, 8193, 8, 0, gray, (CGBitmapInfo)kCGImageAlphaNone);
	CGImageRef image = CGBitmapContextCreateImage(context);
	NSMutableData *png = [NSMutableData data];
	CGImageDestinationRef destination = CGImageDestinationCreateWithData((__bridge CFMutableDataRef)png, CFSTR("public.png"), 1, NULL);
	CGImageDestinationAddImage(destination, image, NULL);
	XCTAssertTrue(CGImageDestinationFinalize(destination));
	CFRelease(destination);
	CGImageRelease(image);
	CGContextRelease(context);
	CGColorSpaceRelease(gray);

	NSError *error = nil;
	XCTAssertNil([[[NFKValueCoding alloc] init] valueOfType:NFKModelValueTypeMask fromFileURL:[self write:png named:@"large.png"] error:&error]);
	[self assertRefusal:error];
	XCTAssertTrue([error.localizedDescription containsString:@"8193"], @"the cap names the declared size: %@", error);
}

- (void)testACMYKImageIsConvertedToRGB
{
	CGColorSpaceRef space = CGColorSpaceCreateDeviceCMYK();
	CGContextRef context = CGBitmapContextCreate(NULL, 4, 4, 8, 16, space, (CGBitmapInfo)kCGImageAlphaNone);
	CGFloat white[5] = { 0, 0, 0, 0, 1 };
	CGContextSetFillColorSpace(context, space);
	CGContextSetFillColor(context, white);
	CGContextFillRect(context, CGRectMake(0, 0, 4, 4));
	CGImageRef image = CGBitmapContextCreateImage(context);
	NSMutableData *tiff = [NSMutableData data];
	CGImageDestinationRef destination = CGImageDestinationCreateWithData((__bridge CFMutableDataRef)tiff, CFSTR("public.tiff"), 1, NULL);
	CGImageDestinationAddImage(destination, image, NULL);
	XCTAssertTrue(CGImageDestinationFinalize(destination));
	CFRelease(destination);
	CGImageRelease(image);
	CGContextRelease(context);
	CGColorSpaceRelease(space);

	NSError *error = nil;
	CVPixelBufferRef buffer = (__bridge CVPixelBufferRef)[[[NFKValueCoding alloc] init] valueOfType:NFKModelValueTypeImage
																						fromFileURL:[self write:tiff named:@"white.tiff"]
																							  error:&error];
	XCTAssertNotEqual(buffer, NULL, @"%@", error);
	CVPixelBufferLockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
	const uint8_t *pixel = CVPixelBufferGetBaseAddress(buffer);
	XCTAssertGreaterThan(pixel[0], 200, @"white comes back white, not its CMYK channels stored as color");
	XCTAssertGreaterThan(pixel[2], 200);
	CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
}

@end
