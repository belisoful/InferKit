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

NSString * const NFKVisionTrackingParameterRegion	= @"trackingRegion";
NSString * const NFKVisionTrackingParameterLevel	= @"trackingLevel";
NSString * const NFKVisionTrackingParameterRestart	= @"restartTracking";

@interface NFKVisionTrackingBackend ()
@property (nonatomic, strong, nullable) VNSequenceRequestHandler *sequence;
@property (nonatomic, strong, nullable) VNDetectedObjectObservation *tracked;
@property (nonatomic, strong, nullable) VNDetectTrajectoriesRequest *trajectories;
@property (nonatomic) NSInteger submittedFrameCount;
/*! The region a request started the sequence from, as the request gave it. */
@property (nonatomic, copy, nullable) NSArray<NSNumber *> *requestedRegion;
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

- (NSSet<NSString *> *)supportedOutputKeys
{
	if (self.kind == NFKVisionTrackingKindTrajectory) {
		return [NSSet setWithObject:NFKOutputStructured];
	}
	return [NSSet setWithObject:NFKOutputDetections];
}

- (NSSet<NSString *> *)supportedParameterKeys
{
	if (self.kind == NFKVisionTrackingKindTrajectory) {
		return [NSSet setWithObject:NFKVisionTrackingParameterRestart];
	}
	return [NSSet setWithObjects:NFKVisionTrackingParameterRegion, NFKVisionTrackingParameterLevel,
			NFKVisionTrackingParameterRestart, nil];
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
	self.requestedRegion = nil;
}

- (void)reset
{
	self.sequence = nil;
	self.tracked = nil;
	self.trajectories = nil;
	self.submittedFrameCount = 0;
	self.requestedRegion = nil;
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError **)outError
{
	NSDictionary<NSString *, id> *parameters = request.parameters;
	NSNumber *level = [self trackingLevelForParameters:parameters error:outError];
	if (level == nil || ![self startSequenceForParameters:parameters error:outError]) {
		return nil;
	}
	CGImageRef image = [NFKVisionSupport imageForRequest:request key:NFKInputImage error:outError];
	if (image == NULL) {
		return nil;
	}
	NFKInferenceResult *result = (self.kind == NFKVisionTrackingKindTrajectory)
		? [self advanceTrajectoriesWithImage:image error:outError]
		: [self advanceTrackingWithImage:image level:level.integerValue error:outError];
	CGImageRelease(image);
	return result;
}

#pragma mark Request parameters

- (nullable NSNumber *)trackingLevelForParameters:(NSDictionary<NSString *, id> *)parameters error:(NSError **)outError
{
	id value = parameters[NFKVisionTrackingParameterLevel];
	if (value == nil) {
		return @(self.trackingLevel);
	}
	NSDictionary<NSString *, NSNumber *> *levels = @{ @"accurate": @(NFKVisionTrackingLevelAccurate),
													  @"fast": @(NFKVisionTrackingLevelFast) };
	NSNumber *level = [value isKindOfClass:NSString.class] ? levels[value] : nil;
	if (level == nil) {
		[self failParameter:NFKVisionTrackingParameterLevel takes:@"\"accurate\" or \"fast\"" error:outError];
	}
	return level;
}

// A region equal to the one the sequence started from continues the sequence, so request defaults that
// repeat the region on every frame follow it.
- (BOOL)startSequenceForParameters:(NSDictionary<NSString *, id> *)parameters error:(NSError **)outError
{
	id restart = parameters[NFKVisionTrackingParameterRestart];
	if (restart != nil && ![restart isKindOfClass:NSNumber.class]) {
		return [self failParameter:NFKVisionTrackingParameterRestart takes:@"a boolean" error:outError];
	}
	id region = parameters[NFKVisionTrackingParameterRegion];
	CGRect box = CGRectNull;
	if (region != nil && ![self region:region asRect:&box]) {
		return [self failParameter:NFKVisionTrackingParameterRegion
							 takes:@"four numbers x, y, width, height, a region inside 0...1" error:outError];
	}
	if ([restart boolValue]) {
		[self reset];
	}
	if (region == nil || (self.isTracking && [region isEqual:self.requestedRegion])) {
		return YES;
	}
	[self startTrackingBoundingBox:box];
	self.requestedRegion = region;
	return YES;
}

- (BOOL)region:(id)value asRect:(CGRect *)rect
{
	if (![value isKindOfClass:NSArray.class] || [value count] != 4) {
		return NO;
	}
	double numbers[4];
	for (NSUInteger index = 0; index < 4; index++) {
		id number = [value objectAtIndex:index];
		if (![number isKindOfClass:NSNumber.class]) {
			return NO;
		}
		numbers[index] = [number doubleValue];
	}
	BOOL inside = numbers[0] >= 0.0 && numbers[1] >= 0.0 && numbers[2] > 0.0 && numbers[3] > 0.0
		&& numbers[0] + numbers[2] <= 1.0 && numbers[1] + numbers[3] <= 1.0;
	if (!inside) {
		return NO;
	}
	*rect = CGRectMake(numbers[0], numbers[1], numbers[2], numbers[3]);
	return YES;
}

- (BOOL)failParameter:(NSString *)key takes:(NSString *)description error:(NSError **)outError
{
	NSString *reason = [NSString stringWithFormat:@"\"%@\" takes %@", key, description];
	return [NFKVisionSupport failWithError:outError code:kNFKError_InferenceUnsupported reason:reason];
}

#pragma mark Object tracking

- (nullable NFKInferenceResult *)advanceTrackingWithImage:(CGImageRef)image
													level:(NFKVisionTrackingLevel)level
													error:(NSError **)outError
{
	VNDetectedObjectObservation *tracked = self.tracked;
	if (tracked == nil || self.sequence == nil) {
		[NFKVisionSupport failWithError:outError
								   code:kNFKError_InferenceMissingInput
								 reason:@"name the region under NFKVisionTrackingParameterRegion or with "
										@"startTrackingBoundingBox: before the first frame"];
		return nil;
	}

	VNTrackObjectRequest *track = [[VNTrackObjectRequest alloc] initWithDetectedObjectObservation:tracked];
	track.trackingLevel = level == NFKVisionTrackingLevelFast ? VNRequestTrackingLevelFast : VNRequestTrackingLevelAccurate;
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
