//
//  NFKVisionMeasurementBackend.m
//  InferKit
//

#import "NFKVisionMeasurementBackend.h"
#import "NFKVisionSupport.h"
#import "NFKInferenceKeys.h"
#import "NFKErrors.h"
#import <Vision/Vision.h>

@implementation NFKVisionMeasurementBackend

+ (instancetype)backend
{
	return [[self alloc] init];
}

+ (instancetype)backendWithKind:(NFKVisionMeasurementKind)kind
{
	NFKVisionMeasurementBackend *backend = [[self alloc] init];
	backend.kind = kind;
	return backend;
}

- (BOOL)isReady
{
	if (self.kind == NFKVisionMeasurementKindAesthetics) {
		if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
			return YES;
		}
		return NO;
	}
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"vision-measurement";
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithObject:NFKInputImage];
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError **)outError
{
	if (![self isReady]) {
		[NFKVisionSupport failWithError:outError
								   code:kNFKError_InferenceUnsupported
								 reason:@"the aesthetics reading needs a newer OS than the one running"];
		return nil;
	}

	CGImageRef image = [NFKVisionSupport imageForRequest:request key:NFKInputImage error:outError];
	if (image == NULL) {
		return nil;
	}

	VNImageBasedRequest *visionRequest = nil;
	if (self.kind == NFKVisionMeasurementKindAesthetics) {
		if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
			visionRequest = [[VNCalculateImageAestheticsScoresRequest alloc] init];
		}
	} else {
		visionRequest = [[VNDetectHorizonRequest alloc] init];
	}
	if (visionRequest == nil) {
		CGImageRelease(image);
		[NFKVisionSupport failWithError:outError
								   code:kNFKError_InferenceUnsupported
								 reason:@"this reading needs a newer OS than the one running"];
		return nil;
	}

	BOOL ran = [NFKVisionSupport performRequests:@[ visionRequest ] onImage:image handler:NULL error:outError];
	CGImageRelease(image);
	if (!ran) {
		return nil;
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputStructured: [self readingFrom:visionRequest] }];
}

- (NSDictionary<NSString *, id> *)readingFrom:(VNImageBasedRequest *)visionRequest
{
	if (self.kind == NFKVisionMeasurementKindAesthetics) {
		if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
			VNImageAestheticsScoresObservation *observation = visionRequest.results.firstObject;
			if (observation == nil) {
				return @{};
			}
			return @{ @"overallScore": @(observation.overallScore),
					  @"isUtility": @(observation.isUtility) };
		}
		return @{};
	}

	// A picture with no visible horizon produces no observation, which is a reading of nothing
	// rather than a failure.
	VNHorizonObservation *observation = visionRequest.results.firstObject;
	if (observation == nil) {
		return @{};
	}
	CGAffineTransform transform = observation.transform;
	return @{ @"angleRadians": @(observation.angle),
			  @"transform": @[ @(transform.a), @(transform.b), @(transform.c),
							   @(transform.d), @(transform.tx), @(transform.ty) ] };
}

@end
