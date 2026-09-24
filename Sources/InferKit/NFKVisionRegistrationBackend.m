//
//  NFKVisionRegistrationBackend.m
//  InferKit
//

#import "NFKVisionRegistrationBackend.h"
#import "NFKVisionSupport.h"
#import "NFKImageCoding.h"
#import "NFKInferenceKeys.h"
#import "NFKErrors.h"
#import <Vision/Vision.h>
#import <simd/simd.h>

@implementation NFKVisionRegistrationBackend

+ (instancetype)backend
{
	return [[self alloc] init];
}

+ (instancetype)backendWithKind:(NFKVisionRegistrationKind)kind
{
	NFKVisionRegistrationBackend *backend = [[self alloc] init];
	backend.kind = kind;
	return backend;
}

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"vision-registration";
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithObject:NFKInputImages];
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError **)outError
{
	NSArray *frames = [request inputForKey:NFKInputImages];
	if (![frames isKindOfClass:NSArray.class] || frames.count < 2) {
		[NFKVisionSupport failWithError:outError
								   code:kNFKError_InferenceMissingInput
								 reason:@"NFKInputImages carries the two frames, the reference one first"];
		return nil;
	}
	CGImageRef reference = [NFKImageCoding CGImageForImage:frames[0]];
	CGImageRef moving = [NFKImageCoding CGImageForImage:frames[1]];
	if (reference == NULL || moving == NULL) {
		CGImageRelease(reference);
		CGImageRelease(moving);
		[NFKVisionSupport failWithError:outError
								   code:kNFKError_InferenceMissingInput
								 reason:@"a frame is not a CGImage, CVPixelBuffer, or BGRA/RGBA texture"];
		return nil;
	}

	// The request is built around the frame that moved and performed against the reference, which is
	// how Vision expresses "align this onto that".
	VNTargetedImageRequest *alignment = nil;
	if (self.kind == NFKVisionRegistrationKindHomographic) {
		alignment = [[VNHomographicImageRegistrationRequest alloc] initWithTargetedCGImage:moving options:@{}];
	} else {
		alignment = [[VNTranslationalImageRegistrationRequest alloc] initWithTargetedCGImage:moving options:@{}];
	}

	BOOL ran = [NFKVisionSupport performRequests:@[ alignment ] onImage:reference handler:NULL error:outError];
	CGImageRelease(reference);
	CGImageRelease(moving);
	if (!ran) {
		return nil;
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputStructured: [self alignmentFrom:alignment] }];
}

- (NSDictionary<NSString *, id> *)alignmentFrom:(VNTargetedImageRequest *)alignment
{
	id observation = alignment.results.firstObject;
	if ([observation isKindOfClass:VNImageTranslationAlignmentObservation.class]) {
		CGAffineTransform transform = ((VNImageTranslationAlignmentObservation *)observation).alignmentTransform;
		return @{ @"transform": @[ @(transform.a), @(transform.b), @(transform.c),
								   @(transform.d), @(transform.tx), @(transform.ty) ] };
	}
	if ([observation isKindOfClass:VNImageHomographicAlignmentObservation.class]) {
		matrix_float3x3 warp = ((VNImageHomographicAlignmentObservation *)observation).warpTransform;
		NSMutableArray<NSNumber *> *numbers = [NSMutableArray arrayWithCapacity:9];
		for (int row = 0; row < 3; row++) {
			for (int column = 0; column < 3; column++) {
				[numbers addObject:@(warp.columns[column][row])];
			}
		}
		return @{ @"warpTransform": numbers };
	}
	return @{};
}

@end
