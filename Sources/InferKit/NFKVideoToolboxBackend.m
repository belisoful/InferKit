//
//  NFKVideoToolboxBackend.m
//  InferKit
//

#import "NFKVideoToolboxBackend.h"
#import "NFKInferenceKeys.h"
#import "NFKInferenceRequest.h"
#import "NFKInferenceResult.h"
#import "NFKImageCoding.h"
#import "NFKErrors.h"
#import <CoreVideo/CoreVideo.h>
#import <VideoToolbox/VideoToolbox.h>

// The frame processors are unavailable on tvOS except interpolation, and every one of them
// postdates the core's floor, so each path is both compiled out where the names do not exist and
// checked at run time where they do.
#if TARGET_OS_TV
	#define NFK_VT_HAS_SUPER_RESOLUTION 0
	#define NFK_VT_HAS_OPTICAL_FLOW 0
#else
	#define NFK_VT_HAS_SUPER_RESOLUTION 1
	#define NFK_VT_HAS_OPTICAL_FLOW 1
#endif

@implementation NFKVideoToolboxBackend

+ (instancetype)backend
{
	return [[self alloc] init];
}

+ (instancetype)backendWithTask:(NFKVideoToolboxTask)task
{
	NFKVideoToolboxBackend *backend = [[self alloc] init];
	backend.task = task;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_scaleFactor = 0;
		_flowScale = 32.0;
	}
	return self;
}

- (NSString *)backendIdentifier
{
	return @"videotoolbox";
}

- (NSSet<NSString *> *)supportedInputKeys
{
	if (self.task == NFKVideoToolboxTaskSuperResolution) {
		return [NSSet setWithObject:NFKInputImage];
	}
	return [NSSet setWithObject:NFKInputImages];
}

- (BOOL)isReady
{
	switch (self.task) {
		case NFKVideoToolboxTaskSuperResolution:
#if NFK_VT_HAS_SUPER_RESOLUTION
			if (@available(macOS 26.0, iOS 26.0, *)) {
				return VTSuperResolutionScalerConfiguration.isSupported;
			}
#endif
			return NO;
		case NFKVideoToolboxTaskFrameInterpolation:
			if (@available(macOS 26.0, iOS 26.0, tvOS 26.0, *)) {
				return VTLowLatencyFrameInterpolationConfiguration.isSupported;
			}
			return NO;
		case NFKVideoToolboxTaskOpticalFlow:
#if NFK_VT_HAS_OPTICAL_FLOW
			if (@available(macOS 15.4, iOS 26.0, *)) {
				return VTOpticalFlowConfiguration.isSupported;
			}
#endif
			return NO;
	}
	return NO;
}

// Upscaling runs a model the system downloads once. Asking for it here, rather than at the first
// run, is what keeps a pending download out of the request path.
- (BOOL)prepareWithError:(NSError **)outError
{
	if (![self isReady]) {
		return [self failWithError:outError
							  code:kNFKError_InferenceUnsupported
							reason:@"this frame processor does not run on this machine or this OS"];
	}
#if NFK_VT_HAS_SUPER_RESOLUTION
	if (self.task == NFKVideoToolboxTaskSuperResolution) {
		if (@available(macOS 26.0, iOS 26.0, *)) {
			NSInteger factor = [self resolvedScaleFactorWithError:outError];
			if (factor == 0) {
				return NO;
			}
			VTSuperResolutionScalerConfiguration *configuration = [self superResolutionConfigurationForWidth:1920
																									  height:1080
																								 scaleFactor:factor];
			if (configuration == nil) {
				return [self failWithError:outError
									  code:kNFKError_InferenceUnsupported
									reason:@"the upscaler rejected a nominal 1920 by 1080 configuration"];
			}
			if (configuration.configurationModelStatus == VTSuperResolutionScalerConfigurationModelStatusReady) {
				return YES;
			}
			if (configuration.configurationModelStatus == VTSuperResolutionScalerConfigurationModelStatusDownloadRequired) {
				[configuration downloadConfigurationModelWithCompletionHandler:^(NSError * _Nullable error) { (void)error; }];
			}
			NSString *reason = [NSString stringWithFormat:@"the upscaling model is still downloading (%.0f%% available)",
								configuration.configurationModelPercentageAvailable * 100.0];
			return [self failWithError:outError code:kNFKError_InferenceNotReady reason:reason];
		}
	}
#endif
	return YES;
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError **)outError
{
	if (![self isReady]) {
		[self failWithError:outError
					   code:kNFKError_InferenceUnsupported
					 reason:@"this frame processor does not run on this machine or this OS"];
		return nil;
	}

	switch (self.task) {
		case NFKVideoToolboxTaskSuperResolution:
			return [self runSuperResolutionForRequest:request error:outError];
		case NFKVideoToolboxTaskFrameInterpolation:
			return [self runInterpolationForRequest:request error:outError];
		case NFKVideoToolboxTaskOpticalFlow:
			return [self runOpticalFlowForRequest:request error:outError];
	}
	[self failWithError:outError code:kNFKError_InferenceUnsupported reason:@"unknown frame-processor task"];
	return nil;
}

#pragma mark Tasks

- (nullable NFKInferenceResult *)runSuperResolutionForRequest:(NFKInferenceRequest *)request
														error:(NSError **)outError
{
#if NFK_VT_HAS_SUPER_RESOLUTION
	if (@available(macOS 26.0, iOS 26.0, *)) {
		CVPixelBufferRef source = [self sourceBufferForRequest:request key:NFKInputImage error:outError];
		if (source == NULL) {
			return nil;
		}
		NSInteger factor = [self resolvedScaleFactorWithError:outError];
		if (factor == 0) {
			CVPixelBufferRelease(source);
			return nil;
		}

		VTSuperResolutionScalerConfiguration *configuration =
			[self superResolutionConfigurationForWidth:(NSInteger)CVPixelBufferGetWidth(source)
												height:(NSInteger)CVPixelBufferGetHeight(source)
										   scaleFactor:factor];
		if (configuration == nil) {
			CVPixelBufferRelease(source);
			[self failWithError:outError
						   code:kNFKError_InferenceUnsupported
						 reason:@"the upscaler does not take this frame size"];
			return nil;
		}
		if (configuration.configurationModelStatus != VTSuperResolutionScalerConfigurationModelStatusReady) {
			CVPixelBufferRelease(source);
			[self failWithError:outError
						   code:kNFKError_InferenceNotReady
						 reason:@"the upscaling model is not downloaded; call prepare first"];
			return nil;
		}

		CVPixelBufferRef reading = [self buffer:source transferredToAttributes:configuration.sourcePixelBufferAttributes];
		CVPixelBufferRef destination = [self bufferMatchingAttributes:configuration.destinationPixelBufferAttributes
															 fallback:source];
		VTFrameProcessorFrame *sourceFrame = [self frameWithBuffer:reading];
		VTFrameProcessorFrame *destinationFrame = [self frameWithBuffer:destination];
		if (reading == NULL || destination == NULL || sourceFrame == nil || destinationFrame == nil) {
			CVPixelBufferRelease(source);
			CVPixelBufferRelease(reading);
			CVPixelBufferRelease(destination);
			[self failWithError:outError code:kNFKError_InferenceBackendFailure reason:@"the frames could not be allocated"];
			return nil;
		}

		VTSuperResolutionScalerParameters *parameters =
			[[VTSuperResolutionScalerParameters alloc] initWithSourceFrame:sourceFrame
															 previousFrame:nil
													   previousOutputFrame:nil
															   opticalFlow:nil
															submissionMode:VTSuperResolutionScalerParametersSubmissionModeRandom
														  destinationFrame:destinationFrame];
		NFKInferenceResult *result = [self runConfiguration:configuration parameters:parameters outputs:nil error:outError];
		if (result != nil) {
			result = [self resultWithBufferConvertedToBGRA:destination error:outError];
		}
		CVPixelBufferRelease(source);
		CVPixelBufferRelease(reading);
		CVPixelBufferRelease(destination);
		return result;
	}
#endif
	[self failWithError:outError code:kNFKError_InferenceUnsupported reason:@"upscaling needs macOS 26 or iOS 26"];
	return nil;
}

- (nullable NFKInferenceResult *)runInterpolationForRequest:(NFKInferenceRequest *)request
													  error:(NSError **)outError
{
	if (@available(macOS 26.0, iOS 26.0, tvOS 26.0, *)) {
		CVPixelBufferRef previous = NULL;
		CVPixelBufferRef source = NULL;
		if (![self framePairForRequest:request previous:&previous source:&source error:outError]) {
			return nil;
		}

		VTLowLatencyFrameInterpolationConfiguration *configuration =
			[[VTLowLatencyFrameInterpolationConfiguration alloc] initWithFrameWidth:(NSInteger)CVPixelBufferGetWidth(source)
																		frameHeight:(NSInteger)CVPixelBufferGetHeight(source)
														 numberOfInterpolatedFrames:1];
		if (configuration == nil) {
			CVPixelBufferRelease(previous);
			CVPixelBufferRelease(source);
			[self failWithError:outError
						   code:kNFKError_InferenceUnsupported
						 reason:@"the interpolator does not take this frame size"];
			return nil;
		}

		CVPixelBufferRef previousReading = [self buffer:previous transferredToAttributes:configuration.sourcePixelBufferAttributes];
		CVPixelBufferRef sourceReading = [self buffer:source transferredToAttributes:configuration.sourcePixelBufferAttributes];
		CVPixelBufferRef middle = [self bufferMatchingAttributes:configuration.destinationPixelBufferAttributes fallback:source];
		// The two frames must differ in presentation time, and the interpolated frame sits between.
		VTFrameProcessorFrame *previousFrame = [self frameWithBuffer:previousReading time:CMTimeMake(0, 600)];
		VTFrameProcessorFrame *sourceFrame = [self frameWithBuffer:sourceReading time:CMTimeMake(600, 600)];
		VTFrameProcessorFrame *middleFrame = [self frameWithBuffer:middle time:CMTimeMake(300, 600)];
		if (previousReading == NULL || sourceReading == NULL || middle == NULL ||
			previousFrame == nil || sourceFrame == nil || middleFrame == nil) {
			CVPixelBufferRelease(previous);
			CVPixelBufferRelease(source);
			CVPixelBufferRelease(previousReading);
			CVPixelBufferRelease(sourceReading);
			CVPixelBufferRelease(middle);
			[self failWithError:outError code:kNFKError_InferenceBackendFailure reason:@"the frames could not be allocated"];
			return nil;
		}

		VTLowLatencyFrameInterpolationParameters *parameters =
			[[VTLowLatencyFrameInterpolationParameters alloc] initWithSourceFrame:sourceFrame
																	previousFrame:previousFrame
															   interpolationPhase:@[ @0.5 ]
																destinationFrames:@[ middleFrame ]];
		NFKInferenceResult *result = [self runConfiguration:configuration parameters:parameters outputs:nil error:outError];
		if (result != nil) {
			result = [self resultWithBufferConvertedToBGRA:middle error:outError];
		}
		CVPixelBufferRelease(previous);
		CVPixelBufferRelease(source);
		CVPixelBufferRelease(previousReading);
		CVPixelBufferRelease(sourceReading);
		CVPixelBufferRelease(middle);
		return result;
	}
	[self failWithError:outError code:kNFKError_InferenceUnsupported reason:@"interpolation needs macOS 26, iOS 26, or tvOS 26"];
	return nil;
}

- (nullable NFKInferenceResult *)runOpticalFlowForRequest:(NFKInferenceRequest *)request
													error:(NSError **)outError
{
#if NFK_VT_HAS_OPTICAL_FLOW
	if (@available(macOS 15.4, iOS 26.0, *)) {
		CVPixelBufferRef previous = NULL;
		CVPixelBufferRef source = NULL;
		if (![self framePairForRequest:request previous:&previous source:&source error:outError]) {
			return nil;
		}

		VTOpticalFlowConfiguration *configuration =
			[[VTOpticalFlowConfiguration alloc] initWithFrameWidth:(NSInteger)CVPixelBufferGetWidth(source)
													   frameHeight:(NSInteger)CVPixelBufferGetHeight(source)
											 qualityPrioritization:VTOpticalFlowConfigurationQualityPrioritizationNormal
														  revision:VTOpticalFlowConfiguration.defaultRevision];
		if (configuration == nil) {
			CVPixelBufferRelease(previous);
			CVPixelBufferRelease(source);
			[self failWithError:outError
						   code:kNFKError_InferenceUnsupported
						 reason:@"the flow estimator does not take this frame size"];
			return nil;
		}

		CVPixelBufferRef previousReading = [self buffer:previous transferredToAttributes:configuration.sourcePixelBufferAttributes];
		CVPixelBufferRef sourceReading = [self buffer:source transferredToAttributes:configuration.sourcePixelBufferAttributes];
		CVPixelBufferRef forward = [self bufferMatchingAttributes:configuration.destinationPixelBufferAttributes fallback:source];
		CVPixelBufferRef backward = [self bufferMatchingAttributes:configuration.destinationPixelBufferAttributes fallback:source];
		VTFrameProcessorFrame *previousFrame = [self frameWithBuffer:previousReading time:CMTimeMake(0, 600)];
		VTFrameProcessorFrame *sourceFrame = [self frameWithBuffer:sourceReading time:CMTimeMake(600, 600)];
		VTFrameProcessorOpticalFlow *flow = nil;
		if (forward != NULL && backward != NULL) {
			flow = [[VTFrameProcessorOpticalFlow alloc] initWithForwardFlow:forward backwardFlow:backward];
		}
		if (flow == nil || previousFrame == nil || sourceFrame == nil) {
			CVPixelBufferRelease(previous);
			CVPixelBufferRelease(source);
			CVPixelBufferRelease(previousReading);
			CVPixelBufferRelease(sourceReading);
			CVPixelBufferRelease(forward);
			CVPixelBufferRelease(backward);
			[self failWithError:outError code:kNFKError_InferenceBackendFailure reason:@"the flow buffers could not be allocated"];
			return nil;
		}

		VTOpticalFlowParameters *parameters =
			[[VTOpticalFlowParameters alloc] initWithSourceFrame:previousFrame
													   nextFrame:sourceFrame
												  submissionMode:VTOpticalFlowParametersSubmissionModeRandom
										  destinationOpticalFlow:flow];
		NFKInferenceResult *result = [self runConfiguration:configuration parameters:parameters outputs:nil error:outError];
		if (result != nil) {
			CVPixelBufferRef packed = [self packedFlowFromBuffer:forward];
			if (packed == NULL) {
				[self failWithError:outError code:kNFKError_InferenceBackendFailure reason:@"the flow field could not be packed"];
				result = nil;
			} else {
				result = [NFKInferenceResult resultWithOutputs:@{ NFKOutputImage: (__bridge id)packed }];
				CVPixelBufferRelease(packed);
			}
		}
		CVPixelBufferRelease(previous);
		CVPixelBufferRelease(source);
		CVPixelBufferRelease(previousReading);
		CVPixelBufferRelease(sourceReading);
		CVPixelBufferRelease(forward);
		CVPixelBufferRelease(backward);
		return result;
	}
#endif
	[self failWithError:outError code:kNFKError_InferenceUnsupported reason:@"optical flow needs macOS 15.4 or iOS 26"];
	return nil;
}

#pragma mark Running

- (nullable NFKInferenceResult *)runConfiguration:(id)configuration
									   parameters:(id)parameters
										  outputs:(nullable NSDictionary<NSString *, id> *)outputs
											error:(NSError **)outError API_AVAILABLE(macos(15.4), ios(26.0), tvos(26.0))
{
	VTFrameProcessor *processor = [[VTFrameProcessor alloc] init];
	NSError *sessionError = nil;
	if (![processor startSessionWithConfiguration:configuration error:&sessionError]) {
		NSString *reason = sessionError.localizedDescription ?: @"the frame processor refused the configuration";
		[self failWithError:outError code:kNFKError_InferenceBackendFailure reason:reason];
		return nil;
	}
	NSError *processError = nil;
	BOOL processed = [processor processWithParameters:parameters error:&processError];
	[processor endSession];
	if (!processed) {
		NSString *reason = processError.localizedDescription ?: @"the frame processor failed";
		[self failWithError:outError code:kNFKError_InferenceBackendFailure reason:reason];
		return nil;
	}
	return [NFKInferenceResult resultWithOutputs:outputs ?: @{}];
}

#pragma mark Frames

#if NFK_VT_HAS_SUPER_RESOLUTION
// The machine publishes the factors its upscaler has; 0 takes the smallest of them, and a factor it
// does not have is refused by name rather than quietly replaced.
- (NSInteger)resolvedScaleFactorWithError:(NSError **)outError API_AVAILABLE(macos(26.0), ios(26.0))
{
	NSArray<NSNumber *> *supported = VTSuperResolutionScalerConfiguration.supportedScaleFactors;
	if (self.scaleFactor <= 0) {
		return supported.firstObject.integerValue;
	}
	for (NSNumber *factor in supported) {
		if (factor.integerValue == self.scaleFactor) {
			return self.scaleFactor;
		}
	}
	NSString *reason = [NSString stringWithFormat:@"this machine upscales by %@, not %ld",
						[supported componentsJoinedByString:@" or "], (long)self.scaleFactor];
	[self failWithError:outError code:kNFKError_InferenceUnsupported reason:reason];
	return 0;
}

- (nullable VTSuperResolutionScalerConfiguration *)superResolutionConfigurationForWidth:(NSInteger)width
																				 height:(NSInteger)height
																			scaleFactor:(NSInteger)scaleFactor API_AVAILABLE(macos(26.0), ios(26.0))
{
	return [[VTSuperResolutionScalerConfiguration alloc]
			   initWithFrameWidth:width
					  frameHeight:height
					  scaleFactor:scaleFactor
						inputType:VTSuperResolutionScalerConfigurationInputTypeImage
			   usePrecomputedFlow:NO
			qualityPrioritization:VTSuperResolutionScalerConfigurationQualityPrioritizationNormal
						 revision:VTSuperResolutionScalerConfiguration.defaultRevision];
}
#endif

- (nullable VTFrameProcessorFrame *)frameWithBuffer:(nullable CVPixelBufferRef)buffer API_AVAILABLE(macos(15.4), ios(26.0), tvos(26.0))
{
	return [self frameWithBuffer:buffer time:kCMTimeZero];
}

- (nullable VTFrameProcessorFrame *)frameWithBuffer:(nullable CVPixelBufferRef)buffer
											   time:(CMTime)time API_AVAILABLE(macos(15.4), ios(26.0), tvos(26.0))
{
	if (buffer == NULL) {
		return nil;
	}
	return [[VTFrameProcessorFrame alloc] initWithBuffer:buffer presentationTimeStamp:time];
}

// A processor writes its own pixel format, so the frame a caller receives is converted back to the
// BGRA every other image key in the contract carries.
- (nullable NFKInferenceResult *)resultWithBufferConvertedToBGRA:(CVPixelBufferRef)buffer
														   error:(NSError **)outError API_AVAILABLE(macos(10.8), ios(16.0), tvos(16.0))
{
	CVPixelBufferRef converted = [self buffer:buffer transferredToBGRA:outError];
	if (converted == NULL) {
		return nil;
	}
	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{ NFKOutputImage: (__bridge id)converted }];
	CVPixelBufferRelease(converted);
	return result;
}

- (nullable CVPixelBufferRef)sourceBufferForRequest:(NFKInferenceRequest *)request
												key:(NSString *)key
											  error:(NSError **)outError CF_RETURNS_RETAINED
{
	id value = [request inputForKey:key];
	if (value == nil) {
		NSString *reason = [NSString stringWithFormat:@"the request carries no frame under %@", key];
		[self failWithError:outError code:kNFKError_InferenceMissingInput reason:reason];
		return NULL;
	}
	return [self bufferForImage:value error:outError];
}

// A pixel buffer passes straight through, which keeps a frame from a capture or decode session on
// its IOSurface; anything else is drawn into a new BGRA buffer.
- (nullable CVPixelBufferRef)bufferForImage:(nullable id)image error:(NSError **)outError CF_RETURNS_RETAINED
{
	if (image == nil) {
		[self failWithError:outError code:kNFKError_InferenceMissingInput reason:@"a frame is missing"];
		return NULL;
	}
	if (CFGetTypeID((__bridge CFTypeRef)image) == CVPixelBufferGetTypeID()) {
		return CVPixelBufferRetain((__bridge CVPixelBufferRef)image);
	}
	CGImageRef cgImage = [NFKImageCoding CGImageForImage:image];
	if (cgImage == NULL) {
		[self failWithError:outError
					   code:kNFKError_InferenceMissingInput
					 reason:@"a frame is not a CVPixelBuffer, CGImage, or BGRA/RGBA texture"];
		return NULL;
	}
	CVPixelBufferRef buffer = [NFKImageCoding pixelBufferWithCGImage:cgImage];
	CGImageRelease(cgImage);
	if (buffer == NULL) {
		[self failWithError:outError code:kNFKError_InferenceBackendFailure reason:@"a frame could not be converted"];
	}
	return buffer;
}

- (BOOL)framePairForRequest:(NFKInferenceRequest *)request
				   previous:(CVPixelBufferRef *)outPrevious
					 source:(CVPixelBufferRef *)outSource
					  error:(NSError **)outError
{
	NSArray *frames = [request inputForKey:NFKInputImages];
	if (![frames isKindOfClass:NSArray.class] || frames.count < 2) {
		return [self failWithError:outError
							  code:kNFKError_InferenceMissingInput
							reason:@"NFKInputImages carries the two frames, the earlier one first"];
	}
	CVPixelBufferRef previous = [self bufferForImage:frames[0] error:outError];
	if (previous == NULL) {
		return NO;
	}
	CVPixelBufferRef source = [self bufferForImage:frames[1] error:outError];
	if (source == NULL) {
		CVPixelBufferRelease(previous);
		return NO;
	}
	*outPrevious = previous;
	*outSource = source;
	return YES;
}

// Each processor states the size and pixel format it reads and writes, and none of them takes the
// BGRA a caller usually has, so every buffer crossing the boundary is transferred into the shape
// the configuration asks for.
- (nullable CVPixelBufferRef)bufferMatchingAttributes:(NSDictionary<NSString *, id> *)attributes
											 fallback:(CVPixelBufferRef)fallback CF_RETURNS_RETAINED
{
	NSNumber *width = attributes[(__bridge NSString *)kCVPixelBufferWidthKey];
	NSNumber *height = attributes[(__bridge NSString *)kCVPixelBufferHeightKey];
	NSMutableDictionary *full = [attributes mutableCopy];
	full[(__bridge NSString *)kCVPixelBufferIOSurfacePropertiesKey] = @{};

	CVPixelBufferRef buffer = NULL;
	size_t bufferWidth = width != nil ? width.unsignedLongValue : CVPixelBufferGetWidth(fallback);
	size_t bufferHeight = height != nil ? height.unsignedLongValue : CVPixelBufferGetHeight(fallback);
	OSType pixelFormat = [self formatInAttributes:attributes fallback:kCVPixelFormatType_32BGRA];
	if (CVPixelBufferCreate(kCFAllocatorDefault, bufferWidth, bufferHeight, pixelFormat,
							(__bridge CFDictionaryRef)full, &buffer) != kCVReturnSuccess) {
		return NULL;
	}
	return buffer;
}

- (nullable CVPixelBufferRef)buffer:(CVPixelBufferRef)source
			 transferredToAttributes:(NSDictionary<NSString *, id> *)attributes CF_RETURNS_RETAINED API_AVAILABLE(macos(10.8), ios(16.0), tvos(16.0))
{
	OSType sourceFormat = CVPixelBufferGetPixelFormatType(source);
	NSNumber *width = attributes[(__bridge NSString *)kCVPixelBufferWidthKey];
	BOOL sameFormat = [self formatInAttributes:attributes fallback:sourceFormat] == sourceFormat;
	BOOL sameSize = width == nil || width.unsignedLongValue == CVPixelBufferGetWidth(source);
	if (sameFormat && sameSize) {
		return CVPixelBufferRetain(source);
	}
	CVPixelBufferRef destination = [self bufferMatchingAttributes:attributes fallback:source];
	if (destination == NULL) {
		return NULL;
	}
	if (![self transferBuffer:source into:destination]) {
		CVPixelBufferRelease(destination);
		return NULL;
	}
	return destination;
}

- (nullable CVPixelBufferRef)buffer:(CVPixelBufferRef)source transferredToBGRA:(NSError **)outError CF_RETURNS_RETAINED API_AVAILABLE(macos(10.8), ios(16.0), tvos(16.0))
{
	NSDictionary *attributes = @{ (__bridge NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
								  (__bridge NSString *)kCVPixelBufferWidthKey: @(CVPixelBufferGetWidth(source)),
								  (__bridge NSString *)kCVPixelBufferHeightKey: @(CVPixelBufferGetHeight(source)),
								  (__bridge NSString *)kCVPixelBufferCGImageCompatibilityKey: @YES };
	CVPixelBufferRef converted = [self buffer:source transferredToAttributes:attributes];
	if (converted == NULL) {
		[self failWithError:outError code:kNFKError_InferenceBackendFailure reason:@"the result could not be converted"];
	}
	return converted;
}

// A configuration states its pixel format as one number or as a list of the formats it takes, and
// the first of a list is the one it prefers.
- (OSType)formatInAttributes:(NSDictionary<NSString *, id> *)attributes fallback:(OSType)fallback
{
	id format = attributes[(__bridge NSString *)kCVPixelBufferPixelFormatTypeKey];
	if ([format isKindOfClass:NSArray.class]) {
		format = ((NSArray *)format).firstObject;
	}
	if ([format isKindOfClass:NSNumber.class]) {
		return (OSType)((NSNumber *)format).unsignedIntValue;
	}
	return fallback;
}

- (BOOL)transferBuffer:(CVPixelBufferRef)source into:(CVPixelBufferRef)destination API_AVAILABLE(macos(10.8), ios(16.0), tvos(16.0))
{
	VTPixelTransferSessionRef session = NULL;
	if (VTPixelTransferSessionCreate(kCFAllocatorDefault, &session) != noErr || session == NULL) {
		return NO;
	}
	VTSessionSetProperty(session, kVTPixelTransferPropertyKey_ScalingMode, kVTScalingMode_Letterbox);
	OSStatus status = VTPixelTransferSessionTransferImage(session, source, destination);
	CFRelease(session);
	return status == noErr;
}

// VideoToolbox writes the field as two half-precision channels of pixel displacement. The packing
// is NFKMLXRAFT's, so a caller reads a flow map the same way whichever engine produced it.
- (nullable CVPixelBufferRef)packedFlowFromBuffer:(CVPixelBufferRef)flow CF_RETURNS_RETAINED
{
	if (CVPixelBufferGetPixelFormatType(flow) != kCVPixelFormatType_TwoComponent16Half) {
		return NULL;
	}
	size_t width = CVPixelBufferGetWidth(flow);
	size_t height = CVPixelBufferGetHeight(flow);
	NSDictionary *attributes = @{ (__bridge NSString *)kCVPixelBufferCGImageCompatibilityKey: @YES,
								  (__bridge NSString *)kCVPixelBufferCGBitmapContextCompatibilityKey: @YES,
								  (__bridge NSString *)kCVPixelBufferIOSurfacePropertiesKey: @{} };
	CVPixelBufferRef packed = NULL;
	if (CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
							(__bridge CFDictionaryRef)attributes, &packed) != kCVReturnSuccess) {
		return NULL;
	}

	CVPixelBufferLockBaseAddress(flow, kCVPixelBufferLock_ReadOnly);
	CVPixelBufferLockBaseAddress(packed, 0);
	const uint8_t *flowBase = CVPixelBufferGetBaseAddress(flow);
	uint8_t *packedBase = CVPixelBufferGetBaseAddress(packed);
	size_t flowStride = CVPixelBufferGetBytesPerRow(flow);
	size_t packedStride = CVPixelBufferGetBytesPerRow(packed);
	double range = 2.0 * (self.flowScale > 0.0 ? self.flowScale : 32.0);
	for (size_t y = 0; y < height; y++) {
		const __fp16 *components = (const __fp16 *)(flowBase + y * flowStride);
		uint8_t *pixels = packedBase + y * packedStride;
		for (size_t x = 0; x < width; x++) {
			double dx = (double)components[2 * x];
			double dy = (double)components[2 * x + 1];
			double red = MIN(MAX(0.5 + dx / range, 0.0), 1.0);
			double green = MIN(MAX(0.5 + dy / range, 0.0), 1.0);
			pixels[4 * x + 0] = 0;							// blue: no third component
			pixels[4 * x + 1] = (uint8_t)lround(green * 255.0);
			pixels[4 * x + 2] = (uint8_t)lround(red * 255.0);
			pixels[4 * x + 3] = 255;
		}
	}
	CVPixelBufferUnlockBaseAddress(packed, 0);
	CVPixelBufferUnlockBaseAddress(flow, kCVPixelBufferLock_ReadOnly);
	return packed;
}

- (BOOL)failWithError:(NSError **)error code:(NSInteger)code reason:(NSString *)reason
{
	if (error != NULL) {
		*error = [NSError errorWithDomain:NFKInferenceErrorDomain
									 code:code
								 userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return NO;
}

@end
