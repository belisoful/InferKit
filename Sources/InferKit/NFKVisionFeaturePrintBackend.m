//
//  NFKVisionFeaturePrintBackend.m
//  InferKit
//

#import "NFKVisionFeaturePrintBackend.h"
#import "NFKVisionSupport.h"
#import "NFKInferenceKeys.h"
#import "NFKErrors.h"
#import <Vision/Vision.h>

@implementation NFKVisionFeaturePrintBackend

+ (instancetype)backend
{
	return [[self alloc] init];
}

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"vision-feature-print";
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

	VNGenerateImageFeaturePrintRequest *print = [[VNGenerateImageFeaturePrintRequest alloc] init];
	BOOL ran = [NFKVisionSupport performRequests:@[ print ] onImage:image handler:NULL error:outError];
	CGImageRelease(image);
	if (!ran) {
		return nil;
	}

	VNFeaturePrintObservation *observation = print.results.firstObject;
	if (observation == nil) {
		[NFKVisionSupport failWithError:outError
								   code:kNFKError_InferenceBackendFailure
								 reason:@"the image produced no feature print"];
		return nil;
	}

	NSArray<NSNumber *> *embedding = [self numbersInObservation:observation];
	if (embedding == nil) {
		[NFKVisionSupport failWithError:outError
								   code:kNFKError_InferenceBackendFailure
								 reason:@"the feature print has an element type the backend does not read"];
		return nil;
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputEmbedding: embedding }];
}

// The print's bytes are float or double by element type; both come back as numbers.
- (nullable NSArray<NSNumber *> *)numbersInObservation:(VNFeaturePrintObservation *)observation
{
	NSData *data = observation.data;
	NSUInteger count = observation.elementCount;
	NSMutableArray<NSNumber *> *numbers = [NSMutableArray arrayWithCapacity:count];
	if (observation.elementType == VNElementTypeFloat) {
		const float *elements = data.bytes;
		for (NSUInteger index = 0; index < count; index++) {
			[numbers addObject:@(elements[index])];
		}
		return numbers;
	}
	if (observation.elementType == VNElementTypeDouble) {
		const double *elements = data.bytes;
		for (NSUInteger index = 0; index < count; index++) {
			[numbers addObject:@(elements[index])];
		}
		return numbers;
	}
	return nil;
}

@end
