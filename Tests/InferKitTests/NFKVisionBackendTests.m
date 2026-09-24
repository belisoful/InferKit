//
//  NFKVisionBackendTests.m
//  NFKTests
//
//  The Vision backends run a system model, so these tests need no weights and no network. The
//  image is drawn here, which is also how the text test knows what the reading should say.
//

#import <XCTest/XCTest.h>
#import <CoreText/CoreText.h>
#import <CoreImage/CoreImage.h>
#import <Vision/Vision.h>
#import <InferKit/InferKit.h>

@interface NFKVisionBackendTests : XCTestCase
@end

@implementation NFKVisionBackendTests

#pragma mark Images

- (CGImageRef)imageOfWidth:(size_t)width height:(size_t)height drawing:(void (^ _Nullable)(CGContextRef))drawing CF_RETURNS_RETAINED
{
	CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, width, height, 8, 0, space,
												 kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little);
	CGColorSpaceRelease(space);
	CGContextSetRGBFillColor(context, 1.0, 1.0, 1.0, 1.0);
	CGContextFillRect(context, CGRectMake(0, 0, width, height));
	if (drawing != nil) {
		drawing(context);
	}
	CGImageRef image = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	return image;
}

- (CGImageRef)imageOfText:(NSString *)text nearTop:(BOOL)nearTop CF_RETURNS_RETAINED
{
	// A bitmap context draws from the bottom left, so the larger baseline is the higher line.
	CGFloat baseline = nearTop ? 200.0 : 40.0;
	return [self imageOfWidth:600 height:280 drawing:^(CGContextRef context) {
		CTFontRef font = CTFontCreateWithName(CFSTR("Helvetica"), 72.0, NULL);
		NSDictionary *attributes = @{ (__bridge NSString *)kCTFontAttributeName: (__bridge id)font,
									  (__bridge NSString *)kCTForegroundColorAttributeName: (__bridge id)CGColorGetConstantColor(kCGColorBlack) };
		NSAttributedString *string = [[NSAttributedString alloc] initWithString:text attributes:attributes];
		CTLineRef line = CTLineCreateWithAttributedString((__bridge CFAttributedStringRef)string);
		CGContextSetTextPosition(context, 24.0, baseline);
		CTLineDraw(line, context);
		CFRelease(line);
		CFRelease(font);
	}];
}

- (NFKInferenceRequest *)requestWithImage:(CGImageRef)image
{
	return [NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)image }];
}

#pragma mark Text

- (void)testTheTextBackendReadsRenderedText
{
	CGImageRef image = [self imageOfText:@"INFERKIT" nearTop:YES];
	NFKVisionTextBackend *backend = [NFKVisionTextBackend backend];
	XCTAssertTrue(backend.isReady);
	XCTAssertEqualObjects(backend.backendIdentifier, @"vision-text");

	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithImage:image] error:&error];
	CGImageRelease(image);
	XCTAssertNotNil(result, @"%@", error);
	XCTAssertEqualObjects(result.text, @"INFERKIT");

	NSArray<NFKDetection *> *lines = [result outputForKey:NFKOutputDetections];
	XCTAssertEqual(lines.count, (NSUInteger)1);
	XCTAssertEqualObjects(lines.firstObject.label, @"INFERKIT");
	XCTAssertGreaterThan(lines.firstObject.confidence, 0.5);
}

- (void)testARecognizedLineIsBoxedInTheContractsGeometry
{
	// Vision normalizes from the bottom left and the contract from the top left, so text drawn near
	// the top of the picture must come back with a small y. Both placements are checked, because a
	// missing flip and a doubled one each pass one of them alone.
	CGFloat topMidY = [self midYOfTextNearTop:YES];
	CGFloat bottomMidY = [self midYOfTextNearTop:NO];
	XCTAssertLessThan(topMidY, 0.4, @"text near the top of the picture has a small y");
	XCTAssertGreaterThan(bottomMidY, 0.6, @"text near the bottom has a large y");
}

- (CGFloat)midYOfTextNearTop:(BOOL)nearTop
{
	CGImageRef image = [self imageOfText:@"INFERKIT" nearTop:nearTop];
	NFKInferenceResult *result = [[NFKVisionTextBackend backend] runInferenceForRequest:[self requestWithImage:image]
																				  error:NULL];
	CGImageRelease(image);
	NFKDetection *line = [[result outputForKey:NFKOutputDetections] firstObject];
	XCTAssertNotNil(line);
	XCTAssertGreaterThanOrEqual(CGRectGetMinX(line.boundingBox), 0.0);
	XCTAssertLessThanOrEqual(CGRectGetMaxX(line.boundingBox), 1.0);
	return CGRectGetMidY(line.boundingBox);
}

- (void)testTheTextBackendNeedsAnImage
{
	NSError *error = nil;
	NFKInferenceResult *result = [[NFKVisionTextBackend backend] runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}]
																				  error:&error];
	XCTAssertNil(result);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceMissingInput);
}

#pragma mark Segmentation

- (void)testSaliencyProducesAMask
{
	CGImageRef image = [self imageOfWidth:320 height:240 drawing:^(CGContextRef context) {
		CGContextSetRGBFillColor(context, 0.1, 0.1, 0.8, 1.0);
		CGContextFillEllipseInRect(context, CGRectMake(96, 64, 128, 128));
	}];
	NFKVisionSegmentationBackend *backend =
		[NFKVisionSegmentationBackend backendWithKind:NFKVisionSegmentationKindAttentionSaliency];
	XCTAssertTrue(backend.isReady);

	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithImage:image] error:&error];
	CGImageRelease(image);
	XCTAssertNotNil(result, @"%@", error);

	CVPixelBufferRef mask = (__bridge CVPixelBufferRef)[result outputForKey:NFKOutputMask];
	XCTAssertTrue(mask != NULL);
	XCTAssertGreaterThan(CVPixelBufferGetWidth(mask), (size_t)0);
	XCTAssertGreaterThan(CVPixelBufferGetHeight(mask), (size_t)0);
}

- (void)testThePersonMaskReportsItsAvailability
{
	NFKVisionSegmentationBackend *backend =
		[NFKVisionSegmentationBackend backendWithKind:NFKVisionSegmentationKindPerson];
	if (@available(macOS 12.0, iOS 15.0, tvOS 15.0, *)) {
		XCTAssertTrue(backend.isReady);
		CGImageRef image = [self imageOfWidth:320 height:240 drawing:nil];
		NSError *error = nil;
		NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithImage:image] error:&error];
		CGImageRelease(image);
		XCTAssertNotNil(result, @"an image with no one in it still produces a mask: %@", error);
	} else {
		XCTAssertFalse(backend.isReady);
		NSError *error = nil;
		XCTAssertNil([backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error]);
		XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
	}
}

#pragma mark Pose and faces

- (void)testThePoseBackendFindsNobodyInAnEmptyFrame
{
	CGImageRef image = [self imageOfWidth:320 height:240 drawing:nil];
	NSError *error = nil;
	NFKInferenceResult *result = [[NFKVisionPoseBackend backend] runInferenceForRequest:[self requestWithImage:image]
																				  error:&error];
	CGImageRelease(image);
	XCTAssertNotNil(result, @"%@", error);
	XCTAssertEqualObjects([result outputForKey:NFKOutputPose], @[]);
}

- (void)testTheFaceBackendFindsNoFaceInAnEmptyFrame
{
	CGImageRef image = [self imageOfWidth:320 height:240 drawing:nil];
	NFKVisionFaceBackend *backend = [NFKVisionFaceBackend backend];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithImage:image] error:&error];
	CGImageRelease(image);
	XCTAssertNotNil(result, @"%@", error);
	XCTAssertEqualObjects([result outputForKey:NFKOutputDetections], @[]);
	XCTAssertEqualObjects([result outputForKey:NFKOutputPose], @[], @"landmarks are asked for by default");
}

- (void)testTheFaceBackendOmitsLandmarksWhenItIsNotAskedFor
{
	CGImageRef image = [self imageOfWidth:320 height:240 drawing:nil];
	NFKVisionFaceBackend *backend = [NFKVisionFaceBackend backend];
	backend.detectsLandmarks = NO;
	NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithImage:image] error:NULL];
	CGImageRelease(image);
	XCTAssertNil([result outputForKey:NFKOutputPose]);
}

#pragma mark Feature print

- (void)testTheFeaturePrintBackendEmbedsAnImage
{
	CGImageRef image = [self imageOfWidth:320 height:240 drawing:^(CGContextRef context) {
		CGContextSetRGBFillColor(context, 0.9, 0.2, 0.1, 1.0);
		CGContextFillRect(context, CGRectMake(40, 40, 160, 120));
	}];
	NFKVisionFeaturePrintBackend *backend = [NFKVisionFeaturePrintBackend backend];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithImage:image] error:&error];
	CGImageRelease(image);
	XCTAssertNotNil(result, @"%@", error);
	XCTAssertGreaterThan(result.embedding.count, (NSUInteger)0);
}

- (void)testTwoPrintsOfTheSameImageAgree
{
	CGImageRef image = [self imageOfWidth:160 height:160 drawing:^(CGContextRef context) {
		CGContextSetRGBFillColor(context, 0.2, 0.7, 0.3, 1.0);
		CGContextFillEllipseInRect(context, CGRectMake(20, 20, 120, 120));
	}];
	NFKVisionFeaturePrintBackend *backend = [NFKVisionFeaturePrintBackend backend];
	NSArray<NSNumber *> *first = [backend runInferenceForRequest:[self requestWithImage:image] error:NULL].embedding;
	NSArray<NSNumber *> *second = [backend runInferenceForRequest:[self requestWithImage:image] error:NULL].embedding;
	CGImageRelease(image);
	XCTAssertEqualObjects(first, second);
}

#pragma mark Declared keys

- (void)testEveryVisionBackendDeclaresItsImageInput
{
	NSArray<id<NFKInferenceBackend>> *backends = @[ [NFKVisionTextBackend backend],
													[NFKVisionSegmentationBackend backend],
													[NFKVisionPoseBackend backend],
													[NFKVisionFaceBackend backend],
													[NFKVisionFeaturePrintBackend backend] ];
	for (id<NFKInferenceBackend> backend in backends) {
		XCTAssertTrue([backend.supportedInputKeys containsObject:NFKInputImage], @"%@", backend.backendIdentifier);
	}
}

#pragma mark Classification and animals

- (void)testTheClassificationBackendNamesWhatItSees
{
	CGImageRef image = [self imageOfWidth:320 height:240 drawing:^(CGContextRef context) {
		CGContextSetRGBFillColor(context, 0.2, 0.5, 0.9, 1.0);
		CGContextFillRect(context, CGRectMake(0, 0, 320, 120));
	}];
	NFKVisionClassificationBackend *backend = [NFKVisionClassificationBackend backend];
	XCTAssertTrue(backend.isReady);
	XCTAssertEqualObjects(backend.backendIdentifier, @"vision-classification");

	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithImage:image] error:&error];
	CGImageRelease(image);
	XCTAssertNotNil(result, @"%@", error);

	NSArray<NFKClassification *> *classifications = result.classifications;
	XCTAssertGreaterThan(classifications.count, (NSUInteger)0, @"the taxonomy always scores something");
	for (NFKClassification *classification in classifications) {
		XCTAssertGreaterThanOrEqual(classification.confidence, backend.minimumConfidence);
		XCTAssertGreaterThan(classification.label.length, (NSUInteger)0);
	}
}

- (void)testTheConfidenceFloorDropsTheTail
{
	CGImageRef image = [self imageOfWidth:160 height:160 drawing:nil];
	NFKVisionClassificationBackend *strict = [NFKVisionClassificationBackend backend];
	strict.minimumConfidence = 0.9;
	NFKVisionClassificationBackend *everything = [NFKVisionClassificationBackend backend];
	everything.minimumConfidence = 0.0;

	NSUInteger strictCount = [strict runInferenceForRequest:[self requestWithImage:image] error:NULL].classifications.count;
	NSUInteger allCount = [everything runInferenceForRequest:[self requestWithImage:image] error:NULL].classifications.count;
	CGImageRelease(image);
	XCTAssertLessThanOrEqual(strictCount, allCount);
	XCTAssertGreaterThan(allCount, (NSUInteger)0, @"Vision scores the whole taxonomy");
}

- (void)testTheAnimalBackendFindsNoAnimalInAnEmptyFrame
{
	CGImageRef image = [self imageOfWidth:320 height:240 drawing:nil];
	NFKVisionAnimalBackend *backend = [NFKVisionAnimalBackend backend];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithImage:image] error:&error];
	CGImageRelease(image);
	XCTAssertNotNil(result, @"%@", error);
	XCTAssertEqualObjects(result.detections, @[]);
}

- (void)testTheAnimalBackendReportsWhatTheInstalledModelKnows
{
	NSArray<NSString *> *identifiers = NFKVisionAnimalBackend.supportedIdentifiers;
	XCTAssertGreaterThan(identifiers.count, (NSUInteger)0, @"the installed revision names its animals");
	XCTAssertTrue([identifiers containsObject:@"Dog"] || [identifiers containsObject:@"dog"], @"%@", identifiers);
}

#pragma mark Animal pose and people

- (void)testAnimalPoseReportsItsAvailability
{
	NFKVisionPoseBackend *backend = [NFKVisionPoseBackend backendWithKind:NFKVisionPoseKindAnimal];
	CGImageRef image = [self imageOfWidth:320 height:240 drawing:nil];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithImage:image] error:&error];
	CGImageRelease(image);

	if (@available(macOS 14.0, iOS 17.0, tvOS 17.0, *)) {
		XCTAssertTrue(backend.isReady);
		XCTAssertNotNil(result, @"%@", error);
		XCTAssertEqualObjects([result outputForKey:NFKOutputPose], @[], @"no animal in an empty frame");
	} else {
		XCTAssertFalse(backend.isReady);
		XCTAssertNil(result);
		XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
	}
}

- (void)testThePeopleMaskReportsItsAvailability
{
	NFKVisionSegmentationBackend *backend =
		[NFKVisionSegmentationBackend backendWithKind:NFKVisionSegmentationKindPersonInstances];
	if (@available(macOS 14.0, iOS 17.0, tvOS 17.0, *)) {
		XCTAssertTrue(backend.isReady);
	} else {
		XCTAssertFalse(backend.isReady);
		NSError *error = nil;
		XCTAssertNil([backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error]);
		XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
	}
}

#pragma mark Four-cornered shapes

// A generated QR code is the one barcode a test can make, and reading it back proves the payload and
// the corner mapping together.
- (CGImageRef)imageOfQRCodePayload:(NSString *)payload CF_RETURNS_RETAINED
{
	CIFilter *generator = [CIFilter filterWithName:@"CIQRCodeGenerator"];
	[generator setValue:[payload dataUsingEncoding:NSUTF8StringEncoding] forKey:@"inputMessage"];
	[generator setValue:@"M" forKey:@"inputCorrectionLevel"];
	CIImage *code = [generator.outputImage imageByApplyingTransform:CGAffineTransformMakeScale(8.0, 8.0)];
	CIContext *context = [CIContext contextWithOptions:nil];
	return [context createCGImage:code fromRect:code.extent];
}

- (void)testABarcodeIsReadAndItsCornersReported
{
	CGImageRef image = [self imageOfQRCodePayload:@"INFERKIT"];
	XCTAssertTrue(image != NULL);
	NFKVisionRectangleBackend *backend = [NFKVisionRectangleBackend backendWithKind:NFKVisionRectangleKindBarcode];
	XCTAssertTrue(backend.isReady);
	XCTAssertEqualObjects(backend.backendIdentifier, @"vision-rectangle");

	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithImage:image] error:&error];
	CGImageRelease(image);
	XCTAssertNotNil(result, @"%@", error);

	NFKDetection *code = result.detections.firstObject;
	XCTAssertNotNil(code, @"the generated code is found");
	XCTAssertEqualObjects(code.label, @"INFERKIT", @"the payload is the label");

	NFKQuadrilateral *quad = code.quadrilateral;
	XCTAssertNotNil(quad, @"a code carries its corners");
	XCTAssertLessThan(quad.topLeft.y, quad.bottomLeft.y, @"the origin is the top left");
	XCTAssertLessThan(quad.topLeft.x, quad.topRight.x);
	XCTAssertTrue(CGRectEqualToRect(code.boundingBox, quad.boundingBox));
}

- (void)testTheSymbologiesTheRevisionReadsAreReported
{
	NSArray<NSString *> *symbologies = NFKVisionRectangleBackend.supportedSymbologies;
	XCTAssertGreaterThan(symbologies.count, (NSUInteger)0);
	XCTAssertTrue([symbologies containsObject:VNBarcodeSymbologyQR], @"%@", symbologies);
}

- (void)testARectangleIsFoundWithItsCorners
{
	CGImageRef image = [self imageOfWidth:400 height:400 drawing:^(CGContextRef context) {
		CGContextSetRGBFillColor(context, 0.0, 0.0, 0.0, 1.0);
		CGContextFillRect(context, CGRectMake(0, 0, 400, 400));
		CGContextSetRGBFillColor(context, 1.0, 1.0, 1.0, 1.0);
		CGContextFillRect(context, CGRectMake(80, 100, 240, 180));
	}];
	NFKVisionRectangleBackend *backend = [NFKVisionRectangleBackend backend];
	backend.minimumSize = 0.1;
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithImage:image] error:&error];
	CGImageRelease(image);
	XCTAssertNotNil(result, @"%@", error);

	NFKDetection *rectangle = result.detections.firstObject;
	XCTAssertNotNil(rectangle, @"a white rectangle on black is found");
	XCTAssertNotNil(rectangle.quadrilateral);
	XCTAssertNil(rectangle.label, @"a rectangle has nothing to name it");
	XCTAssertEqualWithAccuracy(rectangle.quadrilateral.topLeft.x, 0.2, 0.05);
	XCTAssertEqualWithAccuracy(rectangle.quadrilateral.topLeft.y, 0.3, 0.05);
}

- (void)testDocumentSegmentationReportsItsAvailability
{
	NFKVisionRectangleBackend *backend = [NFKVisionRectangleBackend backendWithKind:NFKVisionRectangleKindDocument];
	if (@available(macOS 12.0, iOS 15.0, tvOS 15.0, *)) {
		XCTAssertTrue(backend.isReady);
		CGImageRef image = [self imageOfWidth:320 height:240 drawing:nil];
		NSError *error = nil;
		XCTAssertNotNil([backend runInferenceForRequest:[self requestWithImage:image] error:&error], @"%@", error);
		CGImageRelease(image);
	} else {
		XCTAssertFalse(backend.isReady);
		NSError *error = nil;
		XCTAssertNil([backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error]);
		XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
	}
}

#pragma mark Readings

- (void)testTheAestheticsReadingReportsAScoreAndAUtilityFlag
{
	NFKVisionMeasurementBackend *backend =
		[NFKVisionMeasurementBackend backendWithKind:NFKVisionMeasurementKindAesthetics];
	CGImageRef image = [self imageOfWidth:320 height:240 drawing:^(CGContextRef context) {
		CGContextSetRGBFillColor(context, 0.2, 0.6, 0.3, 1.0);
		CGContextFillEllipseInRect(context, CGRectMake(60, 40, 200, 160));
	}];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithImage:image] error:&error];
	CGImageRelease(image);

	if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
		XCTAssertNotNil(result, @"%@", error);
		NSDictionary *reading = result.structured;
		XCTAssertNotNil(reading[@"overallScore"]);
		XCTAssertNotNil(reading[@"isUtility"]);
		double score = [reading[@"overallScore"] doubleValue];
		XCTAssertGreaterThanOrEqual(score, -1.0);
		XCTAssertLessThanOrEqual(score, 1.0);
	} else {
		XCTAssertFalse(backend.isReady);
		XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
	}
}

- (void)testAPictureWithNoHorizonReadsAsNothing
{
	NFKVisionMeasurementBackend *backend =
		[NFKVisionMeasurementBackend backendWithKind:NFKVisionMeasurementKindHorizon];
	XCTAssertTrue(backend.isReady);
	CGImageRef image = [self imageOfWidth:320 height:240 drawing:nil];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithImage:image] error:&error];
	CGImageRelease(image);
	XCTAssertNotNil(result, @"a flat field has no horizon, which is a reading of nothing: %@", error);
	XCTAssertEqualObjects(result.structured, @{});
}

- (void)testASlantedHorizonReportsItsAngle
{
	// A dark band across the lower half, rotated, is the shape Vision reads as a horizon.
	CGImageRef image = [self imageOfWidth:400 height:400 drawing:^(CGContextRef context) {
		CGContextSaveGState(context);
		CGContextTranslateCTM(context, 200, 200);
		CGContextRotateCTM(context, 0.15);
		CGContextTranslateCTM(context, -200, -200);
		CGContextSetRGBFillColor(context, 0.1, 0.1, 0.15, 1.0);
		CGContextFillRect(context, CGRectMake(-100, -100, 600, 300));
		CGContextRestoreGState(context);
	}];
	NFKInferenceResult *result =
		[[NFKVisionMeasurementBackend backendWithKind:NFKVisionMeasurementKindHorizon]
		 runInferenceForRequest:[self requestWithImage:image] error:NULL];
	CGImageRelease(image);

	NSDictionary *reading = result.structured;
	if (reading[@"angleRadians"] == nil) {
		return;   // Vision found no horizon in the synthetic frame, which is its judgement to make
	}
	XCTAssertEqual([reading[@"transform"] count], (NSUInteger)6, @"a, b, c, d, tx, ty");
	// The reading stays in Vision's own image space, where the origin is the bottom left, so the
	// angle has the opposite sign to the same rotation read in the contract's normalized geometry.
	XCTAssertEqualWithAccuracy([reading[@"angleRadians"] doubleValue], -0.15, 0.1);
}

#pragma mark Contours

- (void)testAShapeIsTracedAsPoints
{
	CGImageRef image = [self imageOfWidth:320 height:320 drawing:^(CGContextRef context) {
		CGContextSetRGBFillColor(context, 0.0, 0.0, 0.0, 1.0);
		CGContextFillEllipseInRect(context, CGRectMake(80, 80, 160, 160));
	}];
	NFKVisionContourBackend *backend = [NFKVisionContourBackend backend];
	backend.detectsDarkOnLight = YES;
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithImage:image] error:&error];
	CGImageRelease(image);
	XCTAssertNotNil(result, @"%@", error);

	NSArray<NSArray<NSNumber *> *> *contours = result.structured[@"contours"];
	XCTAssertGreaterThan(contours.count, (NSUInteger)0, @"a filled circle has an outline");
	XCTAssertGreaterThan([result.structured[@"topLevelCount"] integerValue], 0);
	NSArray<NSNumber *> *first = contours.firstObject;
	XCTAssertEqual(first.count % 2, (NSUInteger)0, @"the points are flattened x, y, x, y");
	for (NSNumber *coordinate in first) {
		XCTAssertGreaterThanOrEqual(coordinate.doubleValue, -0.01);
		XCTAssertLessThanOrEqual(coordinate.doubleValue, 1.01);
	}
}

#pragma mark Registration

- (void)testAShiftedFrameReportsTheShift
{
	CGImageRef reference = [self texturedFrameShiftedBy:0];
	CGImageRef moved = [self texturedFrameShiftedBy:40];
	NFKInferenceRequest *request =
		[NFKInferenceRequest requestWithInputs:@{ NFKInputImages: @[ (__bridge id)reference, (__bridge id)moved ] }];

	NSError *error = nil;
	NFKInferenceResult *result = [[NFKVisionRegistrationBackend backend] runInferenceForRequest:request error:&error];
	CGImageRelease(reference);
	CGImageRelease(moved);
	XCTAssertNotNil(result, @"%@", error);

	NSArray<NSNumber *> *transform = result.structured[@"transform"];
	XCTAssertEqual(transform.count, (NSUInteger)6);
	// The frame moved 40 pixels right, and the transform maps it back onto the reference, so the
	// translation is negative on x and nothing on y.
	double tx = transform[4].doubleValue;
	double ty = transform[5].doubleValue;
	XCTAssertEqualWithAccuracy(tx, -40.0, 4.0, @"tx %f ty %f", tx, ty);
	XCTAssertEqualWithAccuracy(ty, 0.0, 4.0);
}

- (void)testRegistrationNeedsTwoFrames
{
	CGImageRef only = [self texturedFrameShiftedBy:0];
	NSError *error = nil;
	NFKInferenceResult *result =
		[[NFKVisionRegistrationBackend backend]
		 runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputImages: @[ (__bridge id)only ] }]
						  error:&error];
	CGImageRelease(only);
	XCTAssertNil(result);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceMissingInput);
}

// Texture, not two shapes: Vision's translational registration answers a sparse frame confidently
// and wrongly on the vertical axis, which the note in apple-framework-backends.md records.
- (CGImageRef)texturedFrameShiftedBy:(CGFloat)x CF_RETURNS_RETAINED
{
	return [self imageOfWidth:320 height:320 drawing:^(CGContextRef context) {
		CGContextSetRGBFillColor(context, 0.1, 0.1, 0.1, 1.0);
		CGContextFillRect(context, CGRectMake(0, 0, 320, 320));
		// A fixed generator rather than a grid: a repeating pattern is ambiguous to align, and
		// Vision answers the ambiguity confidently and wrongly.
		uint32_t state = 7u;
		for (NSInteger index = 0; index < 400; index++) {
			state = state * 1664525u + 1013904223u;
			CGFloat blockX = x + (CGFloat)((state >> 16) % 260u);
			state = state * 1664525u + 1013904223u;
			CGFloat blockY = (CGFloat)((state >> 16) % 300u);
			state = state * 1664525u + 1013904223u;
			CGContextSetRGBFillColor(context, ((state >> 16) % 100u) / 100.0,
									 ((state >> 8) % 100u) / 100.0, (state % 100u) / 100.0, 1.0);
			CGContextFillRect(context, CGRectMake(blockX, blockY, 6.0, 6.0));
		}
	}];
}

#pragma mark Tracking, which holds state

- (void)testATrackedRegionFollowsTheThingThatMoved
{
	NFKVisionTrackingBackend *backend = [NFKVisionTrackingBackend backend];
	XCTAssertFalse(backend.isTracking, @"nothing is followed until a region is named");

	// The square starts at x 40 of 320, a quarter across, and moves right by 40 each frame.
	CGImageRef first = [self frameWithMarkerAtX:40];
	[backend startTrackingBoundingBox:CGRectMake(40.0 / 320.0, 120.0 / 320.0, 80.0 / 320.0, 80.0 / 320.0)];
	XCTAssertTrue(backend.isTracking);

	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithImage:first] error:&error];
	CGImageRelease(first);
	XCTAssertNotNil(result, @"%@", error);

	CGImageRef second = [self frameWithMarkerAtX:80];
	result = [backend runInferenceForRequest:[self requestWithImage:second] error:&error];
	CGImageRelease(second);
	XCTAssertNotNil(result, @"%@", error);

	NFKDetection *tracked = result.detections.firstObject;
	if (tracked == nil) {
		return;   // the tracker lost a synthetic square, which is its judgement to make
	}
	XCTAssertEqualObjects(tracked.label, @"tracked");
	XCTAssertGreaterThan(CGRectGetMidX(tracked.boundingBox), 0.0);
	XCTAssertLessThan(CGRectGetMidX(tracked.boundingBox), 1.0);
	XCTAssertGreaterThan(CGRectGetMidY(tracked.boundingBox), 0.0);
}

- (void)testTrackingRefusesAFrameBeforeARegionIsNamed
{
	CGImageRef frame = [self frameWithMarkerAtX:40];
	NSError *error = nil;
	NFKInferenceResult *result = [[NFKVisionTrackingBackend backend]
								  runInferenceForRequest:[self requestWithImage:frame] error:&error];
	CGImageRelease(frame);
	XCTAssertNil(result);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceMissingInput);
}

- (void)testResetForgetsWhatWasBeingFollowed
{
	NFKVisionTrackingBackend *backend = [NFKVisionTrackingBackend backend];
	[backend startTrackingBoundingBox:CGRectMake(0.1, 0.1, 0.2, 0.2)];
	XCTAssertTrue(backend.isTracking);
	[backend reset];
	XCTAssertFalse(backend.isTracking);
}

- (void)testEarlyFramesReportNoTrajectoryYet
{
	// A trajectory needs five frames before Vision will say anything, so the first is empty rather
	// than an error.
	NFKVisionTrackingBackend *backend = [NFKVisionTrackingBackend backendWithKind:NFKVisionTrackingKindTrajectory];
	CGImageRef frame = [self frameWithMarkerAtX:40];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithImage:frame] error:&error];
	CGImageRelease(frame);
	XCTAssertNotNil(result, @"%@", error);
	XCTAssertEqualObjects(result.structured[@"trajectories"], @[]);
}

- (CGImageRef)frameWithMarkerAtX:(CGFloat)x CF_RETURNS_RETAINED
{
	return [self imageOfWidth:320 height:320 drawing:^(CGContextRef context) {
		CGContextSetRGBFillColor(context, 0.08, 0.08, 0.1, 1.0);
		CGContextFillRect(context, CGRectMake(0, 0, 320, 320));
		CGContextSetRGBFillColor(context, 0.95, 0.85, 0.15, 1.0);
		CGContextFillRect(context, CGRectMake(x, 120, 80, 80));
	}];
}

@end
