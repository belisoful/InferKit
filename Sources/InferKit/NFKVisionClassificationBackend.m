//
//  NFKVisionClassificationBackend.m
//  InferKit
//

#import "NFKVisionClassificationBackend.h"
#import "NFKVisionSupport.h"
#import "NFKInferenceKeys.h"
#import "NFKClassification.h"
#import "NFKErrors.h"
#import <Vision/Vision.h>

@implementation NFKVisionClassificationBackend

+ (instancetype)backend
{
	return [[self alloc] init];
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_minimumConfidence = 0.1;
	}
	return self;
}

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"vision-classification";
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

	VNClassifyImageRequest *classify = [[VNClassifyImageRequest alloc] init];
	BOOL ran = [NFKVisionSupport performRequests:@[ classify ] onImage:image handler:NULL error:outError];
	CGImageRelease(image);
	if (!ran) {
		return nil;
	}

	NSMutableArray<NFKClassification *> *classifications = [NSMutableArray array];
	for (VNClassificationObservation *observation in classify.results) {
		if (![self keepsObservation:observation]) {
			continue;
		}
		[classifications addObject:[NFKClassification classificationWithLabel:observation.identifier
																   classIndex:classifications.count
																   confidence:observation.confidence]];
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputClassifications: classifications }];
}

// Vision scores the whole taxonomy, so something drops the tail. Its own precision-recall curve is
// the better filter where a caller states which one matters.
- (BOOL)keepsObservation:(VNClassificationObservation *)observation
{
	if (self.minimumRecallForPrecision > 0.0) {
		return [observation hasMinimumRecall:(float)self.minimumRecallForPrecision forPrecision:1.0];
	}
	return observation.confidence >= self.minimumConfidence;
}

@end
