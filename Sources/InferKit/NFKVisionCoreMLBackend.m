//
//  NFKVisionCoreMLBackend.m
//  InferKit
//

#import "NFKVisionCoreMLBackend.h"
#import "NFKVisionSupport.h"
#import "NFKInferenceKeys.h"
#import "NFKClassification.h"
#import "NFKDetection.h"
#import "NFKErrors.h"
#import <CoreML/CoreML.h>
#import <Vision/Vision.h>

@interface NFKVisionCoreMLBackend ()
@property (nonatomic, strong) VNCoreMLModel *visionModel;
@end

@implementation NFKVisionCoreMLBackend

+ (nullable instancetype)backendWithModel:(MLModel *)model error:(NSError **)error
{
	NSError *modelError = nil;
	VNCoreMLModel *visionModel = [VNCoreMLModel modelForMLModel:model error:&modelError];
	if (visionModel == nil) {
		NSString *reason = modelError.localizedDescription ?: @"Vision cannot run this model";
		[NFKVisionSupport failWithError:error code:kNFKError_InferenceUnsupported reason:reason];
		return nil;
	}
	NFKVisionCoreMLBackend *backend = [[self alloc] init];
	backend.visionModel = visionModel;
	return backend;
}

+ (nullable instancetype)backendWithCompiledModelURL:(NSURL *)url error:(NSError **)error
{
	NSError *loadError = nil;
	MLModel *model = [MLModel modelWithContentsOfURL:url error:&loadError];
	if (model == nil) {
		NSString *reason = loadError.localizedDescription ?: @"the compiled model could not be loaded";
		[NFKVisionSupport failWithError:error code:kNFKError_InferenceNotReady reason:reason];
		return nil;
	}
	return [self backendWithModel:model error:error];
}

- (BOOL)isReady
{
	return self.visionModel != nil;
}

- (NSString *)backendIdentifier
{
	return @"vision-coreml";
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithObject:NFKInputImage];
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError **)outError
{
	CGImageRef image = [NFKVisionSupport imageForRequest:request key:NFKInputImage error:outError];
	if (image == NULL) {
		return nil;
	}

	VNCoreMLRequest *modelRequest = [[VNCoreMLRequest alloc] initWithModel:self.visionModel];
	modelRequest.imageCropAndScaleOption = [self cropAndScaleOption];
	BOOL ran = [NFKVisionSupport performRequests:@[ modelRequest ] onImage:image handler:NULL error:outError];
	CGImageRelease(image);
	if (!ran) {
		return nil;
	}
	return [NFKInferenceResult resultWithOutputs:[self outputsFrom:modelRequest.results]];
}

- (VNImageCropAndScaleOption)cropAndScaleOption
{
	switch (self.cropAndScale) {
		case NFKVisionCropAndScaleScaleToFit:	return VNImageCropAndScaleOptionScaleFit;
		case NFKVisionCropAndScaleScaleToFill:	return VNImageCropAndScaleOptionScaleFill;
		case NFKVisionCropAndScaleCenterCrop:	return VNImageCropAndScaleOptionCenterCrop;
	}
	return VNImageCropAndScaleOptionCenterCrop;
}

// The model decides what it emits, so the observation's type decides which key it lands under. A
// model emitting several kinds fills several keys.
- (NSDictionary<NSString *, id> *)outputsFrom:(nullable NSArray<VNObservation *> *)observations
{
	NSMutableArray<NFKClassification *> *classifications = [NSMutableArray array];
	NSMutableArray<NFKDetection *> *detections = [NSMutableArray array];
	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionary];

	for (VNObservation *observation in observations) {
		if ([observation isKindOfClass:VNRecognizedObjectObservation.class]) {
			VNRecognizedObjectObservation *object = (VNRecognizedObjectObservation *)observation;
			VNClassificationObservation *best = object.labels.firstObject;
			[detections addObject:[NFKDetection detectionWithLabel:best.identifier
														classIndex:detections.count
														confidence:best != nil ? best.confidence : object.confidence
													   boundingBox:[NFKVisionSupport contractRect:object.boundingBox]]];
			continue;
		}
		if ([observation isKindOfClass:VNClassificationObservation.class]) {
			VNClassificationObservation *classification = (VNClassificationObservation *)observation;
			if (classification.confidence < self.minimumConfidence) {
				continue;
			}
			[classifications addObject:[NFKClassification classificationWithLabel:classification.identifier
																	   classIndex:classifications.count
																	   confidence:classification.confidence]];
			continue;
		}
		if ([observation isKindOfClass:VNPixelBufferObservation.class]) {
			outputs[NFKOutputMask] = (__bridge id)((VNPixelBufferObservation *)observation).pixelBuffer;
			continue;
		}
		if ([observation isKindOfClass:VNCoreMLFeatureValueObservation.class]) {
			VNCoreMLFeatureValueObservation *feature = (VNCoreMLFeatureValueObservation *)observation;
			id value = feature.featureValue.multiArrayValue ?: feature.featureValue;
			outputs[feature.featureName ?: @"feature"] = value;
		}
	}

	if (classifications.count > 0) {
		outputs[NFKOutputClassifications] = classifications;
	}
	if (detections.count > 0) {
		outputs[NFKOutputDetections] = detections;
	}
	return outputs;
}

@end
