//
//  NFKAppleProviders.m
//  InferKit
//

#import "NFKAppleProviders.h"
#import "NFKVisionTextBackend.h"
#import "NFKVisionSegmentationBackend.h"
#import "NFKVisionPoseBackend.h"
#import "NFKVisionFaceBackend.h"
#import "NFKVisionFeaturePrintBackend.h"
#import "NFKVideoToolboxBackend.h"
#import "NFKSpeechRecognitionBackend.h"

@implementation NFKVisionTextProvider

+ (nullable id<NFKInferenceBackend>)makeInferenceBackend
{
	return [NFKVisionTextBackend backend];
}

@end

@implementation NFKVisionSegmentationProvider

// The subject mask is the one a caller asking for "segmentation" means, and it postdates the core's
// floor; saliency answers everywhere and is the fallback rather than a failure.
+ (nullable id<NFKInferenceBackend>)makeInferenceBackend
{
	NFKVisionSegmentationBackend *subject =
		[NFKVisionSegmentationBackend backendWithKind:NFKVisionSegmentationKindForegroundInstance];
	if (subject.isReady) {
		return subject;
	}
	return [NFKVisionSegmentationBackend backendWithKind:NFKVisionSegmentationKindAttentionSaliency];
}

@end

@implementation NFKVisionPoseProvider

+ (nullable id<NFKInferenceBackend>)makeInferenceBackend
{
	return [NFKVisionPoseBackend backendWithKind:NFKVisionPoseKindBody];
}

@end

@implementation NFKVisionFaceProvider

+ (nullable id<NFKInferenceBackend>)makeInferenceBackend
{
	return [NFKVisionFaceBackend backend];
}

@end

@implementation NFKVisionFeaturePrintProvider

+ (nullable id<NFKInferenceBackend>)makeInferenceBackend
{
	return [NFKVisionFeaturePrintBackend backend];
}

@end

@implementation NFKVideoToolboxUpscalingProvider

// nil where the machine has no such processor, so discovery tries the next provider instead of
// handing back an engine that refuses every request.
+ (nullable id<NFKInferenceBackend>)makeInferenceBackend
{
	NFKVideoToolboxBackend *backend = [NFKVideoToolboxBackend backendWithTask:NFKVideoToolboxTaskSuperResolution];
	return backend.isReady ? backend : nil;
}

@end

@implementation NFKVideoToolboxOpticalFlowProvider

+ (nullable id<NFKInferenceBackend>)makeInferenceBackend
{
	NFKVideoToolboxBackend *backend = [NFKVideoToolboxBackend backendWithTask:NFKVideoToolboxTaskOpticalFlow];
	return backend.isReady ? backend : nil;
}

@end

@implementation NFKSpeechRecognitionProvider

// Returned whether or not the user has allowed recognition yet, because an app asks for that consent
// itself and the backend reports the state through isReady.
+ (nullable id<NFKInferenceBackend>)makeInferenceBackend
{
	return [NFKSpeechRecognitionBackend backend];
}

@end
