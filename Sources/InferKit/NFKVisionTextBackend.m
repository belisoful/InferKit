//
//  NFKVisionTextBackend.m
//  InferKit
//

#import "NFKVisionTextBackend.h"
#import "NFKVisionSupport.h"
#import "NFKInferenceKeys.h"
#import "NFKDetection.h"
#import "NFKErrors.h"
#import <Vision/Vision.h>

@implementation NFKVisionTextBackend

+ (instancetype)backend
{
	return [[self alloc] init];
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_usesAccurateRecognition = YES;
		_correctsLanguage = YES;
	}
	return self;
}

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"vision-text";
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

	VNRecognizeTextRequest *recognize = [[VNRecognizeTextRequest alloc] init];
	recognize.recognitionLevel = self.usesAccurateRecognition ? VNRequestTextRecognitionLevelAccurate
															  : VNRequestTextRecognitionLevelFast;
	recognize.usesLanguageCorrection = self.correctsLanguage;
	if (self.languages.count > 0) {
		recognize.recognitionLanguages = self.languages;
	}
	if (self.customWords.count > 0) {
		recognize.customWords = self.customWords;
	}
	if (self.minimumTextHeight > 0.0) {
		recognize.minimumTextHeight = (float)self.minimumTextHeight;
	}

	BOOL ran = [NFKVisionSupport performRequests:@[ recognize ] onImage:image handler:NULL error:outError];
	CGImageRelease(image);
	if (!ran) {
		return nil;
	}

	NSMutableArray<NSString *> *lines = [NSMutableArray array];
	NSMutableArray<NFKDetection *> *detections = [NSMutableArray array];
	for (VNRecognizedTextObservation *observation in recognize.results) {
		VNRecognizedText *best = [observation topCandidates:1].firstObject;
		if (best == nil) {
			continue;
		}
		[lines addObject:best.string];
		[detections addObject:[NFKDetection detectionWithLabel:best.string
													classIndex:detections.count
													confidence:best.confidence
												   boundingBox:[NFKVisionSupport contractRect:observation.boundingBox]]];
	}

	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: [lines componentsJoinedByString:@"\n"],
													NFKOutputDetections: detections }];
}

@end
