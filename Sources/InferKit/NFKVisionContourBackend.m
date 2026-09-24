//
//  NFKVisionContourBackend.m
//  InferKit
//

#import "NFKVisionContourBackend.h"
#import "NFKVisionSupport.h"
#import "NFKInferenceKeys.h"
#import "NFKErrors.h"
#import <Vision/Vision.h>
#import <simd/simd.h>

@implementation NFKVisionContourBackend

+ (instancetype)backend
{
	return [[self alloc] init];
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_detail = 0.5;
		_contrastAdjustment = 2.0;
	}
	return self;
}

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"vision-contour";
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

	VNDetectContoursRequest *contours = [[VNDetectContoursRequest alloc] init];
	contours.detectsDarkOnLight = self.detectsDarkOnLight;
	contours.contrastAdjustment = (float)self.contrastAdjustment;
	contours.maximumImageDimension = 512;
	BOOL ran = [NFKVisionSupport performRequests:@[ contours ] onImage:image handler:NULL error:outError];
	CGImageRelease(image);
	if (!ran) {
		return nil;
	}

	VNContoursObservation *observation = contours.results.firstObject;
	if (observation == nil) {
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputStructured: @{ @"contours": @[],
																				@"topLevelCount": @0 } }];
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputStructured: [self readingFrom:observation] }];
}

// Vision hands back a tree and a flat index alike; the flat index is what a result can carry, with
// topLevelCount saying how much of it is outermost.
- (NSDictionary<NSString *, id> *)readingFrom:(VNContoursObservation *)observation
{
	NSMutableArray<NSArray<NSNumber *> *> *contours = [NSMutableArray array];
	for (NSInteger index = 0; index < observation.contourCount; index++) {
		VNContour *contour = [observation contourAtIndex:index error:NULL];
		if (contour == nil) {
			continue;
		}
		VNContour *simplified = contour;
		if (self.detail < 1.0) {
			simplified = [contour polygonApproximationWithEpsilon:(float)((1.0 - self.detail) * 0.01)
															error:NULL] ?: contour;
		}
		[contours addObject:[self pointsInContour:simplified]];
	}
	return @{ @"contours": contours, @"topLevelCount": @(observation.topLevelContourCount) };
}

- (NSArray<NSNumber *> *)pointsInContour:(VNContour *)contour
{
	const simd_float2 *points = contour.normalizedPoints;
	NSInteger count = contour.pointCount;
	if (self.maximumPointsPerContour > 0) {
		count = MIN(count, self.maximumPointsPerContour);
	}
	NSMutableArray<NSNumber *> *flattened = [NSMutableArray arrayWithCapacity:(NSUInteger)count * 2];
	for (NSInteger index = 0; index < count; index++) {
		CGPoint point = [NFKVisionSupport contractPoint:CGPointMake(points[index].x, points[index].y)];
		[flattened addObject:@(point.x)];
		[flattened addObject:@(point.y)];
	}
	return flattened;
}

@end
