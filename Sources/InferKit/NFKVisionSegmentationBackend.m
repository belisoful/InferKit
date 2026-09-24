//
//  NFKVisionSegmentationBackend.m
//  InferKit
//

#import "NFKVisionSegmentationBackend.h"
#import "NFKVisionSupport.h"
#import "NFKInferenceKeys.h"
#import "NFKErrors.h"
#import <Vision/Vision.h>

@implementation NFKVisionSegmentationBackend

+ (instancetype)backend
{
	return [[self alloc] init];
}

+ (instancetype)backendWithKind:(NFKVisionSegmentationKind)kind
{
	NFKVisionSegmentationBackend *backend = [[self alloc] init];
	backend.kind = kind;
	return backend;
}

- (BOOL)isReady
{
	return [self isKindAvailable];
}

- (NSString *)backendIdentifier
{
	return @"vision-segmentation";
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithObject:NFKInputImage];
}

// The two newer kinds postdate the core's floor, so readiness is a question the OS answers.
- (BOOL)isKindAvailable
{
	switch (self.kind) {
		case NFKVisionSegmentationKindForegroundInstance:
		case NFKVisionSegmentationKindPersonInstances:
			if (@available(macOS 14.0, iOS 17.0, tvOS 17.0, *)) {
				return YES;
			}
			return NO;
		case NFKVisionSegmentationKindPerson:
			if (@available(macOS 12.0, iOS 15.0, tvOS 15.0, *)) {
				return YES;
			}
			return NO;
		case NFKVisionSegmentationKindAttentionSaliency:
		case NFKVisionSegmentationKindObjectnessSaliency:
			return YES;
	}
	return NO;
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError **)outError
{
	if (![self isKindAvailable]) {
		[NFKVisionSupport failWithError:outError
								   code:kNFKError_InferenceUnsupported
								 reason:@"this mask needs a newer OS than the one running"];
		return nil;
	}

	CGImageRef image = [NFKVisionSupport imageForRequest:request key:NFKInputImage error:outError];
	if (image == NULL) {
		return nil;
	}

	VNImageBasedRequest *visionRequest = [self makeRequest];
	VNImageRequestHandler *handler = nil;
	BOOL ran = [NFKVisionSupport performRequests:@[ visionRequest ] onImage:image handler:&handler error:outError];
	CGImageRelease(image);
	if (!ran) {
		return nil;
	}

	CVPixelBufferRef mask = [self maskFromRequest:visionRequest handler:handler error:outError];
	if (mask == NULL) {
		return nil;
	}
	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{ NFKOutputMask: (__bridge id)mask }];
	CVPixelBufferRelease(mask);
	return result;
}

- (VNImageBasedRequest *)makeRequest
{
	switch (self.kind) {
		case NFKVisionSegmentationKindForegroundInstance:
			if (@available(macOS 14.0, iOS 17.0, tvOS 17.0, *)) {
				return [[VNGenerateForegroundInstanceMaskRequest alloc] init];
			}
			break;
		case NFKVisionSegmentationKindPerson:
			if (@available(macOS 12.0, iOS 15.0, tvOS 15.0, *)) {
				VNGeneratePersonSegmentationRequest *person = [[VNGeneratePersonSegmentationRequest alloc] init];
				person.qualityLevel = [self personQualityLevel];
				return person;
			}
			break;
		case NFKVisionSegmentationKindPersonInstances:
			if (@available(macOS 14.0, iOS 17.0, tvOS 17.0, *)) {
				return [[VNGeneratePersonInstanceMaskRequest alloc] init];
			}
			break;
		case NFKVisionSegmentationKindAttentionSaliency:
			return [[VNGenerateAttentionBasedSaliencyImageRequest alloc] init];
		case NFKVisionSegmentationKindObjectnessSaliency:
			return [[VNGenerateObjectnessBasedSaliencyImageRequest alloc] init];
	}
	return [[VNGenerateAttentionBasedSaliencyImageRequest alloc] init];
}

- (VNGeneratePersonSegmentationRequestQualityLevel)personQualityLevel API_AVAILABLE(macos(12.0), ios(15.0), tvos(15.0))
{
	switch (self.quality) {
		case NFKVisionSegmentationQualityAccurate:	return VNGeneratePersonSegmentationRequestQualityLevelAccurate;
		case NFKVisionSegmentationQualityBalanced:	return VNGeneratePersonSegmentationRequestQualityLevelBalanced;
		case NFKVisionSegmentationQualityFast:		return VNGeneratePersonSegmentationRequestQualityLevelFast;
	}
	return VNGeneratePersonSegmentationRequestQualityLevelAccurate;
}

// The instance mask is generated from the observation and the handler that produced it; the other
// kinds carry their buffer on the observation.
- (nullable CVPixelBufferRef)maskFromRequest:(VNImageBasedRequest *)visionRequest
									 handler:(VNImageRequestHandler *)handler
									   error:(NSError **)outError CF_RETURNS_RETAINED
{
	if (self.kind == NFKVisionSegmentationKindForegroundInstance
		|| self.kind == NFKVisionSegmentationKindPersonInstances) {
		if (@available(macOS 14.0, iOS 17.0, tvOS 17.0, *)) {
			VNInstanceMaskObservation *observation = visionRequest.results.firstObject;
			if (observation == nil) {
				[NFKVisionSupport failWithError:outError
										   code:kNFKError_InferenceBackendFailure
										 reason:@"the image has no subject to mask"];
				return NULL;
			}
			NSError *maskError = nil;
			CVPixelBufferRef mask = [observation generateScaledMaskForImageForInstances:observation.allInstances
																	 fromRequestHandler:handler
																				  error:&maskError];
			if (mask == NULL) {
				NSString *reason = maskError.localizedDescription ?: @"the subject mask could not be generated";
				[NFKVisionSupport failWithError:outError code:kNFKError_InferenceBackendFailure reason:reason];
			}
			return mask;
		}
	}

	VNPixelBufferObservation *observation = visionRequest.results.firstObject;
	if (observation == nil) {
		[NFKVisionSupport failWithError:outError
								   code:kNFKError_InferenceBackendFailure
								 reason:@"the request produced no mask"];
		return NULL;
	}
	return CVPixelBufferRetain(observation.pixelBuffer);
}

@end
