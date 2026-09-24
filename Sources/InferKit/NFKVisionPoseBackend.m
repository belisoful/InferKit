//
//  NFKVisionPoseBackend.m
//  InferKit
//

#import "NFKVisionPoseBackend.h"
#import "NFKVisionSupport.h"
#import "NFKInferenceKeys.h"
#import "NFKKeypoint.h"
#import "NFKErrors.h"
#import <Vision/Vision.h>

@implementation NFKVisionPoseBackend

+ (instancetype)backend
{
	return [[self alloc] init];
}

+ (instancetype)backendWithKind:(NFKVisionPoseKind)kind
{
	NFKVisionPoseBackend *backend = [[self alloc] init];
	backend.kind = kind;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_maximumHandCount = 2;
	}
	return self;
}

- (BOOL)isReady
{
	if (self.kind == NFKVisionPoseKindAnimal) {
		if (@available(macOS 14.0, iOS 17.0, tvOS 17.0, *)) {
			return YES;
		}
		return NO;
	}
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"vision-pose";
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

	if (![self isReady]) {
		CGImageRelease(image);
		[NFKVisionSupport failWithError:outError
								   code:kNFKError_InferenceUnsupported
								 reason:@"animal pose needs a newer OS than the one running"];
		return nil;
	}

	VNImageBasedRequest *poseRequest = nil;
	if (self.kind == NFKVisionPoseKindHand) {
		VNDetectHumanHandPoseRequest *hands = [[VNDetectHumanHandPoseRequest alloc] init];
		hands.maximumHandCount = (NSUInteger)MAX(self.maximumHandCount, 1);
		poseRequest = hands;
	} else if (self.kind == NFKVisionPoseKindAnimal) {
		if (@available(macOS 14.0, iOS 17.0, tvOS 17.0, *)) {
			poseRequest = [[VNDetectAnimalBodyPoseRequest alloc] init];
		}
	} else {
		poseRequest = [[VNDetectHumanBodyPoseRequest alloc] init];
	}
	if (poseRequest == nil) {
		CGImageRelease(image);
		[NFKVisionSupport failWithError:outError
								   code:kNFKError_InferenceUnsupported
								 reason:@"this pose needs a newer OS than the one running"];
		return nil;
	}

	BOOL ran = [NFKVisionSupport performRequests:@[ poseRequest ] onImage:image handler:NULL error:outError];
	CGImageRelease(image);
	if (!ran) {
		return nil;
	}

	NSMutableArray<NFKKeypoint *> *keypoints = [NSMutableArray array];
	for (VNRecognizedPointsObservation *observation in poseRequest.results) {
		[keypoints addObjectsFromArray:[self keypointsInObservation:observation]];
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputPose: keypoints }];
}

// Vision hands back an unordered dictionary, so the joints are sorted by name for a stable index.
- (NSArray<NFKKeypoint *> *)keypointsInObservation:(VNRecognizedPointsObservation *)observation
{
	NSDictionary<VNRecognizedPointKey, VNRecognizedPoint *> *points = [self pointsInObservation:observation];
	NSArray<VNRecognizedPointKey> *names = [points.allKeys sortedArrayUsingSelector:@selector(compare:)];
	NSMutableArray<NFKKeypoint *> *keypoints = [NSMutableArray arrayWithCapacity:names.count];
	for (VNRecognizedPointKey name in names) {
		VNRecognizedPoint *point = points[name];
		[keypoints addObject:[NFKKeypoint keypointWithName:name
													  index:keypoints.count
												   position:[NFKVisionSupport contractPoint:point.location]
												 confidence:point.confidence]];
	}
	return keypoints;
}

- (NSDictionary<VNRecognizedPointKey, VNRecognizedPoint *> *)pointsInObservation:(VNRecognizedPointsObservation *)observation
{
	NSError *pointsError = nil;
	NSDictionary<VNRecognizedPointKey, VNRecognizedPoint *> *points = nil;
	if (@available(macOS 14.0, iOS 17.0, tvOS 17.0, *)) {
		if ([observation isKindOfClass:VNAnimalBodyPoseObservation.class]) {
			VNAnimalBodyPoseObservation *animal = (VNAnimalBodyPoseObservation *)observation;
			points = [animal recognizedPointsForJointsGroupName:VNAnimalBodyPoseObservationJointsGroupNameAll
														  error:&pointsError];
			return points ?: @{};
		}
	}
	if ([observation isKindOfClass:VNHumanHandPoseObservation.class]) {
		VNHumanHandPoseObservation *hand = (VNHumanHandPoseObservation *)observation;
		points = [hand recognizedPointsForJointsGroupName:VNHumanHandPoseObservationJointsGroupNameAll
													error:&pointsError];
	} else if ([observation isKindOfClass:VNHumanBodyPoseObservation.class]) {
		VNHumanBodyPoseObservation *body = (VNHumanBodyPoseObservation *)observation;
		points = [body recognizedPointsForJointsGroupName:VNHumanBodyPoseObservationJointsGroupNameAll
													error:&pointsError];
	}
	return points ?: @{};
}

@end
