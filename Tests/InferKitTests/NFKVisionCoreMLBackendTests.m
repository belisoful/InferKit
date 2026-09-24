//
//  NFKVisionCoreMLBackendTests.m
//  NFKTests
//
//  The models these tests run are written here as Core ML specifications and compiled on the spot,
//  so the backend is exercised end to end with no fixture file. Each model average-pools an 8x8 RGB
//  image and applies one inner product: the "light" score is the mean pixel value times 0.04, less 2,
//  and the "dark" score is 0. A plate whose mean is above about 38 reads as light.
//

#import <XCTest/XCTest.h>
#import <CoreML/CoreML.h>
#import <InferKit/InferKit.h>

#pragma mark Protocol buffer encoding

static void NFKTestAppendVarint(NSMutableData *data, uint64_t value)
{
	do {
		uint8_t byte = value & 0x7F;
		value >>= 7;
		if (value != 0) {
			byte |= 0x80;
		}
		[data appendBytes:&byte length:1];
	} while (value != 0);
}

static void NFKTestAppendKey(NSMutableData *data, uint32_t field, uint32_t wireType)
{
	NFKTestAppendVarint(data, ((uint64_t)field << 3) | wireType);
}

static void NFKTestAppendInteger(NSMutableData *data, uint32_t field, uint64_t value)
{
	NFKTestAppendKey(data, field, 0);
	NFKTestAppendVarint(data, value);
}

static void NFKTestAppendBytes(NSMutableData *data, uint32_t field, NSData *bytes)
{
	NFKTestAppendKey(data, field, 2);
	NFKTestAppendVarint(data, bytes.length);
	[data appendData:bytes];
}

static void NFKTestAppendString(NSMutableData *data, uint32_t field, NSString *string)
{
	NFKTestAppendBytes(data, field, [string dataUsingEncoding:NSUTF8StringEncoding]);
}

static NSData *NFKTestMessage(void (^build)(NSMutableData *message))
{
	NSMutableData *message = [NSMutableData data];
	build(message);
	return message;
}

// The field numbers and enumerated values below are those of Core ML's Model.proto,
// FeatureTypes.proto, and NeuralNetwork.proto.
enum {
	NFKTestSpecificationVersion = 4,
	NFKTestArrayDataTypeDouble = 65600,
	NFKTestColorSpaceRGB = 20,
	NFKTestPoolingAverage = 1,
};

@interface NFKVisionCoreMLBackendTests : XCTestCase
@property (nonatomic, strong) NSMutableArray<NSURL *> *temporaryURLs;
@end

@implementation NFKVisionCoreMLBackendTests

- (void)setUp
{
	self.temporaryURLs = [NSMutableArray array];
}

- (void)tearDown
{
	for (NSURL *url in self.temporaryURLs) {
		[NSFileManager.defaultManager removeItemAtURL:url error:NULL];
	}
}

#pragma mark Model specifications

- (NSData *)featureNamed:(NSString *)name type:(NSData *)type
{
	return NFKTestMessage(^(NSMutableData *feature) {
		NFKTestAppendString(feature, 1, name);
		NFKTestAppendBytes(feature, 3, type);
	});
}

- (NSData *)imageFeatureType
{
	NSData *image = NFKTestMessage(^(NSMutableData *message) {
		NFKTestAppendInteger(message, 1, 8);
		NFKTestAppendInteger(message, 2, 8);
		NFKTestAppendInteger(message, 3, NFKTestColorSpaceRGB);
	});
	return NFKTestMessage(^(NSMutableData *type) { NFKTestAppendBytes(type, 4, image); });
}

- (NSData *)arrayFeatureTypeOfLength:(uint64_t)length
{
	NSData *array = NFKTestMessage(^(NSMutableData *message) {
		NFKTestAppendInteger(message, 1, length);
		NFKTestAppendInteger(message, 2, NFKTestArrayDataTypeDouble);
	});
	return NFKTestMessage(^(NSMutableData *type) { NFKTestAppendBytes(type, 5, array); });
}

- (NSData *)layerNamed:(NSString *)name input:(NSString *)input output:(NSString *)output
				 field:(uint32_t)field parameters:(NSData *)parameters
{
	return NFKTestMessage(^(NSMutableData *layer) {
		NFKTestAppendString(layer, 1, name);
		NFKTestAppendString(layer, 2, input);
		NFKTestAppendString(layer, 3, output);
		NFKTestAppendBytes(layer, field, parameters);
	});
}

- (NSData *)globalAveragePoolFrom:(NSString *)input to:(NSString *)output
{
	NSData *pooling = NFKTestMessage(^(NSMutableData *message) {
		NFKTestAppendInteger(message, 1, NFKTestPoolingAverage);
		NFKTestAppendBytes(message, 30, [NSData data]);
		NFKTestAppendInteger(message, 60, 1);
	});
	return [self layerNamed:@"pool" input:input output:output field:120 parameters:pooling];
}

- (NSData *)brightnessScoresFrom:(NSString *)input to:(NSString *)output
{
	const float third = 0.04f / 3.0f;
	const float weights[] = { 0, 0, 0, third, third, third };
	const float bias[] = { 0, -2 };
	NSData *weightFloats = [NSData dataWithBytes:weights length:sizeof(weights)];
	NSData *biasFloats = [NSData dataWithBytes:bias length:sizeof(bias)];
	NSData *weightValues = NFKTestMessage(^(NSMutableData *message) { NFKTestAppendBytes(message, 1, weightFloats); });
	NSData *biasValues = NFKTestMessage(^(NSMutableData *message) { NFKTestAppendBytes(message, 1, biasFloats); });
	NSData *innerProduct = NFKTestMessage(^(NSMutableData *message) {
		NFKTestAppendInteger(message, 1, 3);
		NFKTestAppendInteger(message, 2, 2);
		NFKTestAppendInteger(message, 10, 1);
		NFKTestAppendBytes(message, 20, weightValues);
		NFKTestAppendBytes(message, 21, biasValues);
	});
	return [self layerNamed:@"scores" input:input output:output field:140 parameters:innerProduct];
}

- (NSData *)modelWithInputs:(NSArray<NSData *> *)inputs
					outputs:(NSArray<NSData *> *)outputs
			 predictedLabel:(nullable NSString *)label
			  probabilities:(nullable NSString *)probabilities
				  typeField:(uint32_t)typeField
					network:(NSData *)network
{
	NSData *description = NFKTestMessage(^(NSMutableData *message) {
		for (NSData *input in inputs) {
			NFKTestAppendBytes(message, 1, input);
		}
		for (NSData *output in outputs) {
			NFKTestAppendBytes(message, 10, output);
		}
		if (label != nil) {
			NFKTestAppendString(message, 11, label);
		}
		if (probabilities != nil) {
			NFKTestAppendString(message, 12, probabilities);
		}
	});
	return NFKTestMessage(^(NSMutableData *model) {
		NFKTestAppendInteger(model, 1, NFKTestSpecificationVersion);
		NFKTestAppendBytes(model, 2, description);
		NFKTestAppendBytes(model, typeField, network);
	});
}

- (NSData *)brightnessClassifierSpecification
{
	NSData *softmax = [self layerNamed:@"softmax" input:@"scores" output:@"probabilities"
								 field:175 parameters:[NSData data]];
	NSData *labels = NFKTestMessage(^(NSMutableData *message) {
		NFKTestAppendString(message, 1, @"dark");
		NFKTestAppendString(message, 1, @"light");
	});
	NSData *network = NFKTestMessage(^(NSMutableData *message) {
		NFKTestAppendBytes(message, 1, [self globalAveragePoolFrom:@"image" to:@"pooled"]);
		NFKTestAppendBytes(message, 1, [self brightnessScoresFrom:@"pooled" to:@"scores"]);
		NFKTestAppendBytes(message, 1, softmax);
		NFKTestAppendBytes(message, 100, labels);
		NFKTestAppendString(message, 200, @"probabilities");
	});
	NSData *stringType = NFKTestMessage(^(NSMutableData *type) { NFKTestAppendBytes(type, 3, [NSData data]); });
	NSData *stringKeys = NFKTestMessage(^(NSMutableData *keys) { NFKTestAppendBytes(keys, 2, [NSData data]); });
	NSData *dictionaryType = NFKTestMessage(^(NSMutableData *type) { NFKTestAppendBytes(type, 6, stringKeys); });
	return [self modelWithInputs:@[ [self featureNamed:@"image" type:[self imageFeatureType]] ]
						 outputs:@[ [self featureNamed:@"label" type:stringType],
									[self featureNamed:@"labelProbability" type:dictionaryType] ]
				  predictedLabel:@"label"
				   probabilities:@"labelProbability"
					   typeField:403
						 network:network];
}

- (NSData *)brightnessScorerSpecification
{
	NSData *network = NFKTestMessage(^(NSMutableData *message) {
		NFKTestAppendBytes(message, 1, [self globalAveragePoolFrom:@"image" to:@"pooled"]);
		NFKTestAppendBytes(message, 1, [self brightnessScoresFrom:@"pooled" to:@"scores"]);
	});
	return [self modelWithInputs:@[ [self featureNamed:@"image" type:[self imageFeatureType]] ]
						 outputs:@[ [self featureNamed:@"scores" type:[self arrayFeatureTypeOfLength:2]] ]
				  predictedLabel:nil
				   probabilities:nil
					   typeField:500
						 network:network];
}

- (NSData *)arrayInputSpecification
{
	NSData *network = NFKTestMessage(^(NSMutableData *message) {
		NFKTestAppendBytes(message, 1, [self brightnessScoresFrom:@"values" to:@"scores"]);
	});
	return [self modelWithInputs:@[ [self featureNamed:@"values" type:[self arrayFeatureTypeOfLength:3]] ]
						 outputs:@[ [self featureNamed:@"scores" type:[self arrayFeatureTypeOfLength:2]] ]
				  predictedLabel:nil
				   probabilities:nil
					   typeField:500
						 network:network];
}

- (NSURL *)compiledModelFromSpecification:(NSData *)specification
{
	NSURL *source = [[NSURL fileURLWithPath:NSTemporaryDirectory()]
					 URLByAppendingPathComponent:[NSUUID.UUID.UUIDString stringByAppendingPathExtension:@"mlmodel"]];
	[self.temporaryURLs addObject:source];
	XCTAssertTrue([specification writeToURL:source atomically:YES]);
	NSError *error = nil;
	NSURL *compiled = [MLModel compileModelAtURL:source error:&error];
	XCTAssertNotNil(compiled, @"%@", error);
	if (compiled != nil) {
		[self.temporaryURLs addObject:compiled];
	}
	return compiled;
}

#pragma mark Images

- (CGImageRef)plateOfWidth:(size_t)width height:(size_t)height whiteFromColumn:(size_t)firstWhite CF_RETURNS_RETAINED
{
	CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, width, height, 8, 0, space,
												 kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little);
	CGColorSpaceRelease(space);
	CGContextSetRGBFillColor(context, 0.0, 0.0, 0.0, 1.0);
	CGContextFillRect(context, CGRectMake(0, 0, width, height));
	CGContextSetRGBFillColor(context, 1.0, 1.0, 1.0, 1.0);
	CGContextFillRect(context, CGRectMake(firstWhite, 0, width - firstWhite, height));
	CGImageRef image = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	return image;
}

- (nullable NFKInferenceResult *)runBackend:(NFKVisionCoreMLBackend *)backend
									onPlate:(CGImageRef)plate
									  error:(NSError **)error
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)plate }];
	NFKInferenceResult *result = [backend runInferenceForRequest:request error:error];
	CGImageRelease(plate);
	return result;
}

- (NFKVisionCoreMLBackend *)classifier
{
	NSError *error = nil;
	NSURL *compiled = [self compiledModelFromSpecification:[self brightnessClassifierSpecification]];
	NFKVisionCoreMLBackend *backend = [NFKVisionCoreMLBackend backendWithCompiledModelURL:compiled error:&error];
	XCTAssertNotNil(backend, @"%@", error);
	return backend;
}

#pragma mark Construction

- (void)testABackendOverACompiledModelIsReady
{
	NFKVisionCoreMLBackend *backend = [self classifier];
	XCTAssertTrue(backend.isReady);
	XCTAssertEqualObjects(backend.backendIdentifier, @"vision-coreml");
	XCTAssertEqualObjects(backend.supportedInputKeys, [NSSet setWithObject:NFKInputImage]);
	XCTAssertEqual(backend.cropAndScale, NFKVisionCropAndScaleCenterCrop, @"Vision's own default");
	XCTAssertEqual(backend.minimumConfidence, 0.0);
}

- (void)testALoadedModelBuildsTheSameBackend
{
	NSError *error = nil;
	NSURL *compiled = [self compiledModelFromSpecification:[self brightnessClassifierSpecification]];
	MLModel *model = [MLModel modelWithContentsOfURL:compiled error:&error];
	XCTAssertNotNil(model, @"%@", error);
	NFKVisionCoreMLBackend *backend = [NFKVisionCoreMLBackend backendWithModel:model error:&error];
	XCTAssertTrue(backend.isReady, @"%@", error);
}

- (void)testAMissingCompiledModelIsNotReady
{
	NSError *error = nil;
	NFKVisionCoreMLBackend *backend =
		[NFKVisionCoreMLBackend backendWithCompiledModelURL:[NSURL fileURLWithPath:@"/nonexistent/model.mlmodelc"]
													  error:&error];
	XCTAssertNil(backend);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceNotReady);
}

- (void)testAModelWhoseInputIsNotAnImageIsRefused
{
	NSError *error = nil;
	NSURL *compiled = [self compiledModelFromSpecification:[self arrayInputSpecification]];
	NFKVisionCoreMLBackend *backend = [NFKVisionCoreMLBackend backendWithCompiledModelURL:compiled error:&error];
	XCTAssertNil(backend);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
}

#pragma mark Running

- (void)testAClassifierNamesWhatItSees
{
	NFKVisionCoreMLBackend *backend = [self classifier];
	NSError *error = nil;
	NFKInferenceResult *white = [self runBackend:backend onPlate:[self plateOfWidth:8 height:8 whiteFromColumn:0] error:&error];
	XCTAssertEqualObjects(white.classifications.firstObject.label, @"light", @"%@", error);
	NFKInferenceResult *black = [self runBackend:backend onPlate:[self plateOfWidth:8 height:8 whiteFromColumn:8] error:&error];
	XCTAssertEqualObjects(black.classifications.firstObject.label, @"dark", @"%@", error);
	XCTAssertEqual(black.classifications.count, (NSUInteger)2, @"no floor keeps every label");
	XCTAssertGreaterThanOrEqual(black.classifications.firstObject.confidence,
								black.classifications.lastObject.confidence, @"best first");
}

- (void)testTheConfidenceFloorDropsTheTail
{
	NFKVisionCoreMLBackend *backend = [self classifier];
	backend.minimumConfidence = 0.9;
	NSError *error = nil;
	NFKInferenceResult *result = [self runBackend:backend onPlate:[self plateOfWidth:8 height:8 whiteFromColumn:0] error:&error];
	XCTAssertEqual(result.classifications.count, (NSUInteger)1, @"%@", error);
	XCTAssertEqualObjects(result.classifications.firstObject.label, @"light");
}

- (void)testTheCropDecidesWhatTheModelSees
{
	// Black except the right third. A centre crop sees only black; scaling to fill sees the whole
	// plate, whose mean clears the threshold.
	NFKVisionCoreMLBackend *backend = [self classifier];
	NSError *error = nil;
	NFKInferenceResult *cropped = [self runBackend:backend onPlate:[self plateOfWidth:24 height:8 whiteFromColumn:16] error:&error];
	XCTAssertEqualObjects(cropped.classifications.firstObject.label, @"dark", @"%@", error);

	backend.cropAndScale = NFKVisionCropAndScaleScaleToFill;
	NFKInferenceResult *filled = [self runBackend:backend onPlate:[self plateOfWidth:24 height:8 whiteFromColumn:16] error:&error];
	XCTAssertEqualObjects(filled.classifications.firstObject.label, @"light", @"%@", error);
}

- (void)testAnArrayOutputArrivesUnderItsFeatureName
{
	NSError *error = nil;
	NSURL *compiled = [self compiledModelFromSpecification:[self brightnessScorerSpecification]];
	NFKVisionCoreMLBackend *backend = [NFKVisionCoreMLBackend backendWithCompiledModelURL:compiled error:&error];
	XCTAssertNotNil(backend, @"%@", error);

	NFKInferenceResult *result = [self runBackend:backend onPlate:[self plateOfWidth:8 height:8 whiteFromColumn:0] error:&error];
	MLMultiArray *scores = [result outputForKey:@"scores"];
	XCTAssertTrue([scores isKindOfClass:MLMultiArray.class], @"%@ %@", result.outputs, error);
	XCTAssertEqual(scores.count, (NSInteger)2);
	XCTAssertEqualWithAccuracy(scores[1].doubleValue, 255.0 * 0.04 - 2.0, 1e-3, @"a white plate's light score");
	XCTAssertNil(result.classifications, @"a model with no labels classifies nothing");
}

- (void)testARequestWithoutAnImageIsRefused
{
	NFKVisionCoreMLBackend *backend = [self classifier];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error];
	XCTAssertNil(result);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceMissingInput);
}

@end
