//
//  NFKCoreMLBackend.m
//  InferKit
//

#import "NFKCoreMLBackend.h"
#import "NFKInferenceRequest.h"
#import "NFKInferenceResult.h"
#import "NFKErrors.h"
#import "NFK_ARC.h"
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>

/*! Maps an IOSurface FourCC pixel format to the Metal format for wrapping it as a texture. */
static MTLPixelFormat NFKMTLPixelFormatForIOSurface(IOSurfaceRef surface)
{
	OSType fourCC = IOSurfaceGetPixelFormat(surface);
	switch (fourCC) {
		case kCVPixelFormatType_64RGBAHalf:		return MTLPixelFormatRGBA16Float;
		case kCVPixelFormatType_128RGBAFloat:	return MTLPixelFormatRGBA32Float;
		case kCVPixelFormatType_32BGRA:			return MTLPixelFormatBGRA8Unorm;
		default:								return MTLPixelFormatRGBA8Unorm;
	}
}

@implementation NFKCoreMLBackend
{
	MLModel *_model;
	id<MTLDevice> _device;
}

@synthesize modelURL = _modelURL;
@synthesize computeUnits = _computeUnits;

+ (instancetype)backendWithModelURL:(nullable NSURL *)modelURL
{
	NFKCoreMLBackend *backend = [[self alloc] init];
	backend.modelURL = modelURL;
	return NARC_AUTORELEASE(backend);
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		// MLComputeUnitsCPUOnly is zero, so leaving this unset would quietly move every model to the
		// CPU. Core ML's own default is All.
		_computeUnits = MLComputeUnitsAll;
	}
	return self;
}

- (void)dealloc
{
	NARC_RELEASE(_model);
	NARC_RELEASE(_device);
	NARC_RELEASE(_modelURL);
	SUPER_DEALLOC();
}

#pragma mark NFKInferenceBackend

- (BOOL)isReady
{
	return _model != nil;
}

- (NSString *)backendIdentifier
{
	return @"coreml";
}

- (BOOL)prepareWithError:(NSError * _Nullable *)outError
{
	if (_model != nil) {
		return YES;
	}
	if (self.modelURL == nil) {
		[self setError:outError code:kNFKError_InferenceNotReady reason:@"no model URL is set"];
		return NO;
	}
	return [self loadModelFromURL:self.modelURL error:outError];
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
													error:(NSError * _Nullable *)outError
{
	if (_model == nil && ![self prepareWithError:outError]) {
		return nil;
	}

	MLModelDescription *description = _model.modelDescription;
	NSMutableDictionary<NSString *, MLFeatureValue *> *features =
		[NSMutableDictionary dictionaryWithCapacity:description.inputDescriptionsByName.count];
	for (NSString *name in description.inputDescriptionsByName) {
		id value = [request inputForKey:name];
		if (value == nil) {
			value = [request parameterForKey:name];
		}
		if (value == nil) {
			NSString *reason = [NSString stringWithFormat:@"the model input '%@' is not in the request", name];
			[self setError:outError code:kNFKError_InferenceMissingInput reason:reason];
			return nil;
		}
		MLFeatureValue *feature = [self featureValueForObject:value
												 description:description.inputDescriptionsByName[name]
													   error:outError];
		if (feature == nil) {
			return nil;
		}
		features[name] = feature;
	}

	NSError *error = nil;
	MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:features
																							   error:&error];
	if (provider == nil) {
		[self propagateError:error to:outError];
		return nil;
	}
	id<MLFeatureProvider> prediction = [_model predictionFromFeatures:provider error:&error];
	NARC_RELEASE(provider);
	if (prediction == nil) {
		[self propagateError:error to:outError];
		return nil;
	}

	NSMutableDictionary<NSString *, id> *outputs =
		[NSMutableDictionary dictionaryWithCapacity:description.outputDescriptionsByName.count];
	for (NSString *name in description.outputDescriptionsByName) {
		id object = [self objectForFeatureValue:[prediction featureValueForName:name]];
		if (object != nil) {
			outputs[name] = object;
		}
	}
	return [NFKInferenceResult resultWithOutputs:outputs];
}

#pragma mark Model loading

- (BOOL)loadModelFromURL:(NSURL *)url error:(NSError * _Nullable *)outError
{
	if (url == nil) {
		[self setError:outError code:kNFKError_InferenceNotReady reason:@"no model URL to load"];
		return NO;
	}

	NSError *error = nil;
	NSURL *compiledURL = url;
	NSString *extension = url.pathExtension.lowercaseString;
	if ([extension isEqualToString:@"mlpackage"] || [extension isEqualToString:@"mlmodel"]) {
		compiledURL = [MLModel compileModelAtURL:url error:&error];
		if (compiledURL == nil) {
			[self propagateError:error to:outError];
			return NO;
		}
	}

	MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
	configuration.computeUnits = self.computeUnits;
	MLModel *model = [MLModel modelWithContentsOfURL:compiledURL configuration:configuration error:&error];
	NARC_RELEASE(configuration);
	if (model == nil) {
		[self propagateError:error to:outError];
		return NO;
	}

	NARC_RELEASE(_model);
	_model = NARC_RETAIN(model);
	self.modelURL = url;
	return YES;
}

#pragma mark Feature conversion

- (nullable MLFeatureValue *)featureValueForObject:(id)value
									   description:(MLFeatureDescription *)featureDescription
											 error:(NSError * _Nullable *)outError
{
	if ([value isKindOfClass:MLFeatureValue.class]) {
		return value;
	}
	if ([value conformsToProtocol:@protocol(MTLTexture)]) {
		id<MTLTexture> texture = (id<MTLTexture>)value;
		if (_device == nil) {
			_device = NARC_RETAIN(texture.device);
		}
		CVPixelBufferRef pixelBuffer = [self pixelBufferFromTexture:texture];
		if (pixelBuffer == NULL) {
			[self setError:outError
					  code:kNFKError_InferenceBackendFailure
					reason:@"the source texture is not IOSurface-backed"];
			return nil;
		}
		MLFeatureValue *feature = [MLFeatureValue featureValueWithPixelBuffer:pixelBuffer];
		CVPixelBufferRelease(pixelBuffer);
		return feature;
	}
	if (CFGetTypeID((__bridge CFTypeRef)value) == CVPixelBufferGetTypeID()) {
		return [MLFeatureValue featureValueWithPixelBuffer:(__bridge CVPixelBufferRef)value];
	}
	if ([value isKindOfClass:MLMultiArray.class]) {
		return [MLFeatureValue featureValueWithMultiArray:value];
	}
	if ([value isKindOfClass:NSNumber.class]) {
		if (featureDescription.type == MLFeatureTypeInt64) {
			return [MLFeatureValue featureValueWithInt64:[value longLongValue]];
		}
		return [MLFeatureValue featureValueWithDouble:[value doubleValue]];
	}
	if ([value isKindOfClass:NSString.class]) {
		return [MLFeatureValue featureValueWithString:value];
	}
	[self setError:outError
			  code:kNFKError_InferenceBackendFailure
			reason:[NSString stringWithFormat:@"unsupported input type %@", [value class]]];
	return nil;
}

- (nullable id)objectForFeatureValue:(nullable MLFeatureValue *)feature
{
	if (feature == nil) {
		return nil;
	}
	switch (feature.type) {
		case MLFeatureTypeImage: {
			id<MTLTexture> texture = [self textureFromPixelBuffer:feature.imageBufferValue];
			return texture != nil ? (id)texture : (__bridge id)feature.imageBufferValue;
		}
		case MLFeatureTypeMultiArray:	return feature.multiArrayValue;
		case MLFeatureTypeDouble:		return @(feature.doubleValue);
		case MLFeatureTypeInt64:		return @(feature.int64Value);
		case MLFeatureTypeString:		return feature.stringValue;
		case MLFeatureTypeDictionary:	return feature.dictionaryValue;
		default:					return nil;
	}
}

- (nullable CVPixelBufferRef)pixelBufferFromTexture:(id<MTLTexture>)texture CF_RETURNS_RETAINED
{
	IOSurfaceRef surface = texture.iosurface;
	if (surface == NULL) {
		return NULL;
	}
	CVPixelBufferRef pixelBuffer = NULL;
	if (CVPixelBufferCreateWithIOSurface(kCFAllocatorDefault, surface, NULL, &pixelBuffer) != kCVReturnSuccess) {
		return NULL;
	}
	return pixelBuffer;
}

- (nullable id<MTLTexture>)textureFromPixelBuffer:(nullable CVPixelBufferRef)pixelBuffer
{
	if (pixelBuffer == NULL) {
		return nil;
	}
	IOSurfaceRef surface = CVPixelBufferGetIOSurface(pixelBuffer);
	if (surface == NULL) {
		return nil;
	}
	id<MTLDevice> device = _device;
	BOOL createdDevice = NO;
	if (device == nil) {
		device = MTLCreateSystemDefaultDevice();
		createdDevice = YES;
	}
	if (device == nil) {
		return nil;
	}
	MTLTextureDescriptor *descriptor =
		[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:NFKMTLPixelFormatForIOSurface(surface)
														   width:IOSurfaceGetWidth(surface)
														  height:IOSurfaceGetHeight(surface)
													   mipmapped:NO];
	descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
	id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor iosurface:surface plane:0];
	if (createdDevice) {
		NARC_RELEASE(device);
	}
	return NARC_AUTORELEASE(texture);
}

#pragma mark Errors

- (BOOL)setError:(NSError * _Nullable *)outError code:(NSInteger)code reason:(NSString *)reason
{
	if (outError != NULL) {
		*outError = [NSError errorWithDomain:NFKInferenceErrorDomain
										code:code
									userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return NO;
}

- (BOOL)propagateError:(nullable NSError *)error to:(NSError * _Nullable *)outError
{
	if (outError == NULL) {
		return NO;
	}
	if (error != nil) {
		*outError = error;
	} else {
		[self setError:outError code:kNFKError_InferenceBackendFailure reason:@"the Core ML run failed"];
	}
	return NO;
}

@end
