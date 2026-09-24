//
//  NFKVisionTrackingBackend.m
//  InferKit
//

#import "NFKVisionTrackingBackend.h"
#import "NFKVisionSupport.h"
#import "NFKInferenceKeys.h"
#import "NFKDetection.h"
#import "NFKErrors.h"
#import "NFKImageCoding.h"
#import <CoreMedia/CoreMedia.h>
#import <Vision/Vision.h>
#import <simd/simd.h>

@interface NFKVisionTrackingBackend ()
@property (nonatomic, strong, nullable) VNSequenceRequestHandler *sequence;
@property (nonatomic, strong, nullable) VNDetectedObjectObservation *tracked;
@property (nonatomic, strong, nullable) VNDetectTrajectoriesRequest *trajectories;
@property (nonatomic) NSInteger submittedFrameCount;
@end

@implementation NFKVisionTrackingBackend

+ (instancetype)backend
{
	return [[self alloc] init];
}

+ (instancetype)backendWithKind:(NFKVisionTrackingKind)kind
{
	NFKVisionTrackingBackend *backend = [[self alloc] init];
	backend.kind = kind;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_trajectoryLength = 5;
		_framesPerSecond = 30.0;
	}
	return self;
}

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"vision-tracking";
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithObject:NFKInputImage];
}

- (BOOL)isTracking
{
	return self.tracked != nil;
}

- (void)startTrackingBoundingBox:(CGRect)boundingBox
{
	// The caller names the region in the contract's geometry; Vision wants its own.
	CGRect visionBox = [NFKVisionSupport contractRect:boundingBox];
	self.sequence = [[VNSequenceRequestHandler alloc] init];
	self.tracked = [VNDetectedObjectObservation observationWithBoundingBox:visionBox];
	self.trajectories = nil;
}

- (void)reset
{
	self.sequence = nil;
	self.tracked = nil;
	self.trajectories = nil;
	self.submittedFrameCount = 0;
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError **)outError
{
	CGImageRef image = [NFKVisionSupport imageForRequest:request key:NFKInputImage error:outError];
	if (image == NULL) {
		return nil;
	}
	NFKInferenceResult *result = (self.kind == NFKVisionTrackingKindTrajectory)
		? [self advanceTrajectoriesWithImage:image error:outError]
		: [self advanceTrackingWithImage:image error:outError];
	CGImageRelease(image);
	return result;
}

#pragma mark Object tracking

- (nullable NFKInferenceResult *)advanceTrackingWithImage:(CGImageRef)image error:(NSError **)outError
{
	VNDetectedObjectObservation *tracked = self.tracked;
	if (tracked == nil || self.sequence == nil) {
		[NFKVisionSupport failWithError:outError
								   code:kNFKError_InferenceMissingInput
								 reason:@"name the region with startTrackingBoundingBox: before the first frame"];
		return nil;
	}

	VNTrackObjectRequest *track = [[VNTrackObjectRequest alloc] initWithDetectedObjectObservation:tracked];
	track.trackingLevel = VNRequestTrackingLevelAccurate;
	NSError *trackError = nil;
	if (![self.sequence performRequests:@[ track ] onCGImage:image error:&trackError]) {
		NSString *reason = trackError.localizedDescription ?: @"the tracker failed on this frame";
		[NFKVisionSupport failWithError:outError code:kNFKError_InferenceBackendFailure reason:reason];
		return nil;
	}

	VNDetectedObjectObservation *observation = track.results.firstObject;
	if (observation == nil) {
		// The region is gone. The sequence ends here rather than following nothing.
		self.tracked = nil;
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputDetections: @[] }];
	}
	self.tracked = observation;

	NFKDetection *detection = [NFKDetection detectionWithLabel:@"tracked"
													classIndex:0
													confidence:observation.confidence
												   boundingBox:[NFKVisionSupport contractRect:observation.boundingBox]];
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputDetections: @[ detection ] }];
}

#pragma mark Trajectories

- (nullable NFKInferenceResult *)advanceTrajectoriesWithImage:(CGImageRef)image error:(NSError **)outError
{
	if (self.sequence == nil) {
		self.sequence = [[VNSequenceRequestHandler alloc] init];
	}
	if (self.trajectories == nil) {
		self.trajectories =
			[[VNDetectTrajectoriesRequest alloc] initWithFrameAnalysisSpacing:kCMTimeZero
															trajectoryLength:MAX(self.trajectoryLength, 5)
														   completionHandler:nil];
	}

	// Vision times a trajectory, and a still image carries no timestamp, so the backend stamps each
	// frame itself at framesPerSecond in the order they arrive.
	CMSampleBufferRef sample = [self timedSampleForImage:image];
	if (sample == NULL) {
		[NFKVisionSupport failWithError:outError
								   code:kNFKError_InferenceBackendFailure
								 reason:@"the frame could not be timed for trajectory detection"];
		return nil;
	}
	NSError *trajectoryError = nil;
	BOOL ran = [self.sequence performRequests:@[ self.trajectories ] onCMSampleBuffer:sample error:&trajectoryError];
	CFRelease(sample);
	if (!ran) {
		NSString *reason = trajectoryError.localizedDescription ?: @"the trajectory detector failed on this frame";
		[NFKVisionSupport failWithError:outError code:kNFKError_InferenceBackendFailure reason:reason];
		return nil;
	}

	NSMutableArray<NSDictionary<NSString *, id> *> *found = [NSMutableArray array];
	for (VNTrajectoryObservation *observation in self.trajectories.results) {
		simd_float3 coefficients = observation.equationCoefficients;
		[found addObject:@{ @"detectedPoints": [self flattenedPoints:observation.detectedPoints],
							@"projectedPoints": [self flattenedPoints:observation.projectedPoints],
							@"equationCoefficients": @[ @(coefficients[0]), @(coefficients[1]), @(coefficients[2]) ],
							@"confidence": @(observation.confidence) }];
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputStructured: @{ @"trajectories": found } }];
}

- (nullable CMSampleBufferRef)timedSampleForImage:(CGImageRef)image CF_RETURNS_RETAINED
{
	CVPixelBufferRef buffer = [NFKImageCoding pixelBufferWithCGImage:image];
	if (buffer == NULL) {
		return NULL;
	}
	CMVideoFormatDescriptionRef format = NULL;
	if (CMVideoFormatDescriptionCreateForImageBuffer(kCFAllocatorDefault, buffer, &format) != noErr) {
		CVPixelBufferRelease(buffer);
		return NULL;
	}

	double rate = self.framesPerSecond > 0.0 ? self.framesPerSecond : 30.0;
	CMTimeScale scale = 600;
	CMSampleTimingInfo timing = {
		.duration = CMTimeMakeWithSeconds(1.0 / rate, scale),
		.presentationTimeStamp = CMTimeMakeWithSeconds(self.submittedFrameCount / rate, scale),
		.decodeTimeStamp = kCMTimeInvalid,
	};
	CMSampleBufferRef sample = NULL;
	OSStatus status = CMSampleBufferCreateForImageBuffer(kCFAllocatorDefault, buffer, true, NULL, NULL,
														 format, &timing, &sample);
	CFRelease(format);
	CVPixelBufferRelease(buffer);
	if (status != noErr) {
		return NULL;
	}
	self.submittedFrameCount += 1;
	return sample;
}

- (NSArray<NSNumber *> *)flattenedPoints:(NSArray<VNPoint *> *)points
{
	NSMutableArray<NSNumber *> *flattened = [NSMutableArray arrayWithCapacity:points.count * 2];
	for (VNPoint *point in points) {
		CGPoint inContract = [NFKVisionSupport contractPoint:CGPointMake(point.x, point.y)];
		[flattened addObject:@(inContract.x)];
		[flattened addObject:@(inContract.y)];
	}
	return flattened;
}

@end
