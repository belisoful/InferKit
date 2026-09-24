//
//  NFKVisionAnimalBackend.m
//  InferKit
//

#import "NFKVisionAnimalBackend.h"
#import "NFKVisionSupport.h"
#import "NFKInferenceKeys.h"
#import "NFKDetection.h"
#import "NFKErrors.h"
#import <Vision/Vision.h>

@implementation NFKVisionAnimalBackend

+ (instancetype)backend
{
	return [[self alloc] init];
}

+ (NSArray<NSString *> *)supportedIdentifiers
{
	VNRecognizeAnimalsRequest *request = [[VNRecognizeAnimalsRequest alloc] init];
	NSError *error = nil;
	return [request supportedIdentifiersAndReturnError:&error] ?: @[];
}

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"vision-animal";
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

	VNRecognizeAnimalsRequest *animals = [[VNRecognizeAnimalsRequest alloc] init];
	BOOL ran = [NFKVisionSupport performRequests:@[ animals ] onImage:image handler:NULL error:outError];
	CGImageRelease(image);
	if (!ran) {
		return nil;
	}

	NSMutableArray<NFKDetection *> *detections = [NSMutableArray array];
	for (VNRecognizedObjectObservation *observation in animals.results) {
		// An observation carries its labels ordered by confidence; the first is the animal.
		VNClassificationObservation *best = observation.labels.firstObject;
		[detections addObject:[NFKDetection detectionWithLabel:best.identifier
													classIndex:detections.count
													confidence:best != nil ? best.confidence : observation.confidence
												   boundingBox:[NFKVisionSupport contractRect:observation.boundingBox]]];
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputDetections: detections }];
}

@end
