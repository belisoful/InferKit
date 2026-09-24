//
//  NFKVisionFaceBackend.m
//  InferKit
//

#import "NFKVisionFaceBackend.h"
#import "NFKVisionSupport.h"
#import "NFKInferenceKeys.h"
#import "NFKDetection.h"
#import "NFKKeypoint.h"
#import "NFKErrors.h"
#import <Vision/Vision.h>

@implementation NFKVisionFaceBackend

+ (instancetype)backend
{
	return [[self alloc] init];
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_detectsLandmarks = YES;
	}
	return self;
}

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"vision-face";
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

	VNImageBasedRequest *faceRequest = self.detectsLandmarks ? [[VNDetectFaceLandmarksRequest alloc] init]
															 : [[VNDetectFaceRectanglesRequest alloc] init];
	BOOL ran = [NFKVisionSupport performRequests:@[ faceRequest ] onImage:image handler:NULL error:outError];
	CGImageRelease(image);
	if (!ran) {
		return nil;
	}

	NSMutableArray<NFKDetection *> *faces = [NSMutableArray array];
	NSMutableArray<NFKKeypoint *> *landmarks = [NSMutableArray array];
	for (VNFaceObservation *observation in faceRequest.results) {
		[faces addObject:[NFKDetection detectionWithLabel:@"face"
											   classIndex:faces.count
											   confidence:observation.confidence
											  boundingBox:[NFKVisionSupport contractRect:observation.boundingBox]]];
		[landmarks addObjectsFromArray:[self landmarksInObservation:observation]];
	}

	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionaryWithObject:faces
																					  forKey:NFKOutputDetections];
	if (self.detectsLandmarks) {
		outputs[NFKOutputPose] = landmarks;
	}
	return [NFKInferenceResult resultWithOutputs:outputs];
}

// A landmark is normalized inside the face's box, so each point is mapped into the image before the
// origin is flipped to the contract's.
- (NSArray<NFKKeypoint *> *)landmarksInObservation:(VNFaceObservation *)observation
{
	VNFaceLandmarkRegion2D *region = observation.landmarks.allPoints;
	const CGPoint *points = region.normalizedPoints;
	if (points == NULL) {
		return @[];
	}
	CGRect box = observation.boundingBox;
	NSMutableArray<NFKKeypoint *> *landmarks = [NSMutableArray arrayWithCapacity:region.pointCount];
	for (NSUInteger index = 0; index < region.pointCount; index++) {
		CGPoint inImage = CGPointMake(box.origin.x + points[index].x * box.size.width,
									  box.origin.y + points[index].y * box.size.height);
		[landmarks addObject:[NFKKeypoint keypointWithName:nil
													  index:(NSInteger)index
												   position:[NFKVisionSupport contractPoint:inImage]
												 confidence:observation.confidence]];
	}
	return landmarks;
}

@end
