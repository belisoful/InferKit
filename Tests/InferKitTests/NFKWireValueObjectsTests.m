//
//  NFKWireValueObjectsTests.m
//  InferKitTests
//
//  The JSON form of InferKit's value classes: each round-trips through JSON text, the default coder
//  keeps its keyed archives, and the decoder reads the JSON form whatever the setting.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
#import "../../Sources/InferKit/NFKInferenceWireCoding.h"
#import "../../Sources/InferKit/NFKWireValueObjects.h"
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>

@interface NFKWireValueObjectsTests : XCTestCase
@end

@implementation NFKWireValueObjectsTests

- (NFKQuadrilateral *)quadrilateral
{
	return [NFKQuadrilateral quadrilateralWithTopLeft:CGPointMake(0.25, 0.75) topRight:CGPointMake(0.75, 0.75)
										   bottomLeft:CGPointMake(0.25, 0.25) bottomRight:CGPointMake(0.75, 0.25)];
}

/*! One value of every class the JSON form covers, with figures JSON carries exactly. */
- (NSArray *)values
{
	NFKMIDINote *note = [[NFKMIDINote alloc] initWithPitch:60 startSeconds:0.5 endSeconds:1.25 velocity:90 program:4
												percussion:NO pitchBend:@[ @0, @0.5 ]];
	NSDictionary *raw = @{ @"type": @"choice", @"choice": @"spam" };
	return @[
		[[NFKDetection alloc] initWithLabel:@"cat" classIndex:3 confidence:0.875 boundingBox:CGRectMake(0.125, 0.25, 0.5, 0.5)
							  quadrilateral:[self quadrilateral]],
		[[NFKDetection alloc] initWithLabel:nil classIndex:-1 confidence:0.5 boundingBox:CGRectMake(0, 0, 1, 1) quadrilateral:nil],
		[[NFKKeypoint alloc] initWithName:@"left_wrist" index:9 position:CGPointMake(0.5, 0.25) confidence:0.75],
		[[NFKClassification alloc] initWithLabel:@"dog" classIndex:2 confidence:0.625],
		[[NFKAudioSegment alloc] initWithStartSeconds:1.5 endSeconds:2.25 label:@"hello" confidence:0.5 speaker:@"A"],
		note,
		[[NFKMIDISequence alloc] initWithNotes:@[ note ] tempoBPM:96 beatsPerBar:3 beatUnit:4 ticksPerQuarterNote:960],
		[[NFKMusicBeat alloc] initWithTimeSeconds:2.5 positionInBar:1],
		[self quadrilateral],
		[[NFKDecisionAnswer alloc] initWithType:NFKDecisionTypeChoice choice:@"spam" score:0 probability:0 confidence:0.5
								  probabilities:@{ @"spam": @0.75, @"ham": @0.25 } legend:nil raw:raw],
		[NFKDecisionAnswer refusalForType:NFKDecisionTypeNoul raw:@{ @"refusal": @"no" }],
	];
}

- (nullable id)roundTrip:(id)value coder:(NFKInferenceWireCoding *)coder
{
	NSError *error = nil;
	id encoded = [coder JSONValueForValue:value key:@"value" error:&error];
	XCTAssertNotNil(encoded, @"%@", error);
	NSData *text = [NSJSONSerialization dataWithJSONObject:@[ encoded ] options:0 error:&error];
	XCTAssertNotNil(text, @"%@", error);
	id parsed = [NSJSONSerialization JSONObjectWithData:text options:0 error:NULL];
	return [coder valueFromJSONValue:[parsed firstObject] key:@"value" error:&error];
}

- (void)testEveryValueClassRoundTripsThroughItsJSONForm
{
	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	coder.encodesValueObjectsAsJSON = YES;
	for (id value in [self values]) {
		NSDictionary *encoded = [coder JSONValueForValue:value key:@"value" error:NULL];
		XCTAssertNotEqualObjects(encoded[@"$nfk"], @"archive", @"%@", value);
		XCTAssertTrue(NFKWireIsValueObjectTag(encoded[@"$nfk"]), @"%@", encoded);
		XCTAssertEqualObjects([self roundTrip:value coder:coder], value, @"%@", encoded);
	}
}

- (void)testADetectionsJSONFormNamesItsFields
{
	NFKDetection *detection = [[NFKDetection alloc] initWithLabel:@"cat" classIndex:3 confidence:0.875
													  boundingBox:CGRectMake(0.125, 0.25, 0.5, 0.5) quadrilateral:nil];
	NSDictionary *expected = @{ @"$nfk": @"detection", @"label": @"cat", @"classIndex": @3, @"confidence": @0.875,
								@"boundingBox": @{ @"x": @0.125, @"y": @0.25, @"width": @0.5, @"height": @0.5 } };
	XCTAssertEqualObjects(NFKWireJSONForValueObject(detection), expected);

	NFKKeypoint *keypoint = [[NFKKeypoint alloc] initWithName:@"nose" index:0 position:CGPointMake(0.5, 0.25) confidence:1];
	NSDictionary *expectedKeypoint = @{ @"$nfk": @"keypoint", @"name": @"nose", @"index": @0,
										@"position": @{ @"x": @0.5, @"y": @0.25 }, @"confidence": @1 };
	XCTAssertEqualObjects(NFKWireJSONForValueObject(keypoint), expectedKeypoint);
}

- (void)testAnArrayOfValueObjectsEncodesEachElement
{
	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	coder.encodesValueObjectsAsJSON = YES;
	NSArray *pose = @[ @[ [[NFKKeypoint alloc] initWithName:@"nose" index:0 position:CGPointMake(0.5, 0.5) confidence:1] ] ];
	NSArray *encoded = [coder JSONValueForValue:pose key:NFKOutputPose error:NULL];
	XCTAssertEqualObjects(encoded[0][0][@"$nfk"], @"keypoint");
	XCTAssertEqualObjects([self roundTrip:pose coder:coder], pose);
}

- (void)testANonFiniteFigureIsNull
{
	NFKClassification *classification = [[NFKClassification alloc] initWithLabel:@"x" classIndex:0 confidence:NAN];
	NSDictionary *encoded = NFKWireJSONForValueObject(classification);
	XCTAssertEqualObjects(encoded[@"confidence"], NSNull.null);
	XCTAssertNotNil([NSJSONSerialization dataWithJSONObject:encoded options:0 error:NULL]);
	NFKClassification *decoded = NFKWireValueObjectFromJSON(@"classification", encoded, NULL);
	XCTAssertTrue(isnan(decoded.confidence));
}

- (void)testTheDefaultCoderKeepsItsArchives
{
	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	XCTAssertFalse(coder.encodesValueObjectsAsJSON);
	for (id value in [self values]) {
		NSDictionary *encoded = [coder JSONValueForValue:value key:@"value" error:NULL];
		XCTAssertEqualObjects(encoded[@"$nfk"], @"archive", @"%@", value);
		XCTAssertEqualObjects([[encoded allKeys] sortedArrayUsingSelector:@selector(compare:)], (@[ @"$nfk", @"base64" ]));
	}
}

- (void)testTheDecoderReadsTheJSONFormWithoutTheSetting
{
	NFKInferenceWireCoding *writer = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	writer.encodesValueObjectsAsJSON = YES;
	NFKInferenceWireCoding *reader = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	for (id value in [self values]) {
		id encoded = [writer JSONValueForValue:value key:@"value" error:NULL];
		XCTAssertEqualObjects([reader valueFromJSONValue:encoded key:@"value" error:NULL], value);
	}
}

- (void)testANumberKeyedDictionaryRoundTripsWithItsKeys
{
	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	NSDictionary *probabilities = @{ @3: @0.75, @7: @0.25, @0.5: @"half" };
	NSDictionary *encoded = [coder JSONValueForValue:probabilities key:@"classProbability" error:NULL];
	XCTAssertEqualObjects(encoded[@"$nfk"], @"numberKeyedDictionary");
	XCTAssertEqualObjects(encoded[@"value"][@"3"], @0.75);
	XCTAssertEqualObjects([self roundTrip:probabilities coder:coder], probabilities);

	NSError *error = nil;
	XCTAssertNil([coder JSONValueForValue:(@{ @1: @1, @"two": @2 }) key:@"mixed" error:&error], @"mixed keys have no form");
	XCTAssertNotNil(error);
}

- (void)testACoreMLFeatureValueEncodesWhatItHolds
{
	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	XCTAssertEqualObjects([coder JSONValueForValue:[MLFeatureValue featureValueWithInt64:42] key:@"f" error:NULL], @42);
	XCTAssertEqualObjects([coder JSONValueForValue:[MLFeatureValue featureValueWithString:@"cat"] key:@"f" error:NULL], @"cat");
	MLFeatureValue *dictionary = [MLFeatureValue featureValueWithDictionary:@{ @1: @0.5 } error:NULL];
	XCTAssertEqualObjects([[coder JSONValueForValue:dictionary key:@"f" error:NULL] objectForKey:@"$nfk"], @"numberKeyedDictionary");
	MLFeatureValue *sequence = [MLFeatureValue featureValueWithSequence:[MLSequence sequenceWithStringArray:@[ @"a", @"b" ]]];
	XCTAssertEqualObjects([coder JSONValueForValue:sequence key:@"f" error:NULL], (@[ @"a", @"b" ]));
}

- (void)testAFloatTextureCrossesAsAPixelBufferOfItsFormat
{
	id<MTLDevice> device = MTLCreateSystemDefaultDevice();
	if (device == nil) {
		XCTSkip(@"this machine has no Metal device");
	}
	MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
																						  width:4 height:2 mipmapped:NO];
	descriptor.storageMode = MTLStorageModeShared;
	id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
	float pixels[2 * 4 * 4];
	for (NSUInteger index = 0; index < 32; index++) {
		pixels[index] = (float)index * 0.25f;
	}
	[texture replaceRegion:MTLRegionMake2D(0, 0, 4, 2) mipmapLevel:0 withBytes:pixels bytesPerRow:4 * 4 * sizeof(float)];

	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	NSError *error = nil;
	NSDictionary *encoded = [coder JSONValueForValue:texture key:@"depth" error:&error];
	XCTAssertEqualObjects(encoded[@"$nfk"], @"pixelBuffer", @"%@", error);
	XCTAssertEqualObjects(encoded[@"format"], @(kCVPixelFormatType_128RGBAFloat));
	id decoded = [coder valueFromJSONValue:encoded key:@"depth" error:&error];
	XCTAssertNotNil(decoded, @"%@", error);
	CVPixelBufferRef buffer = (__bridge CVPixelBufferRef)decoded;
	CVPixelBufferLockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
	const uint8_t *base = CVPixelBufferGetBaseAddress(buffer);
	size_t row = CVPixelBufferGetBytesPerRow(buffer);
	XCTAssertEqual(((const float *)(base + row))[5], pixels[21]);
	CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
}

- (void)testAValueObjectMissingARequiredFieldFails
{
	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	NSArray<NSDictionary *> *malformed = @[
		@{ @"$nfk": @"detection", @"label": @"cat" },
		@{ @"$nfk": @"keypoint", @"position": @{ @"x": @1 } },
		@{ @"$nfk": @"quadrilateral", @"topLeft": @{ @"x": @0, @"y": @0 } },
		@{ @"$nfk": @"midiSequence", @"notes": @[ @{ @"pitch": @60 } ] },
		@{ @"$nfk": @"decisionAnswer", @"choice": @"spam" },
	];
	for (NSDictionary *entry in malformed) {
		NSError *error = nil;
		XCTAssertNil([coder valueFromJSONValue:entry key:@"value" error:&error], @"%@", entry);
		XCTAssertEqualObjects(error.domain, NFKInferenceErrorDomain, @"%@", entry);
	}
}

@end
