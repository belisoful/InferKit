//
//  NFKVisionSupport.m
//  InferKit
//

#import "NFKVisionSupport.h"
#import "NFKInferenceRequest.h"
#import "NFKImageCoding.h"
#import "NFKErrors.h"

@implementation NFKVisionSupport

+ (nullable CGImageRef)imageForRequest:(NFKInferenceRequest *)request
								   key:(NSString *)key
								 error:(NSError **)error
{
	id value = [request inputForKey:key];
	if (value == nil) {
		NSString *reason = [NSString stringWithFormat:@"the request carries no image under %@", key];
		[self failWithError:error code:kNFKError_InferenceMissingInput reason:reason];
		return NULL;
	}
	CGImageRef image = [NFKImageCoding CGImageForImage:value];
	if (image == NULL) {
		NSString *reason = [NSString stringWithFormat:@"the image under %@ is not a CGImage, CVPixelBuffer, or BGRA/RGBA texture", key];
		[self failWithError:error code:kNFKError_InferenceMissingInput reason:reason];
		return NULL;
	}
	return image;
}

+ (BOOL)performRequests:(NSArray<VNRequest *> *)requests
				onImage:(CGImageRef)image
				handler:(VNImageRequestHandler * _Nullable * _Nullable)outHandler
				  error:(NSError **)error
{
	VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:image options:@{}];
	NSError *visionError = nil;
	if (![handler performRequests:requests error:&visionError]) {
		NSString *reason = visionError.localizedDescription ?: @"the Vision request failed";
		return [self failWithError:error code:kNFKError_InferenceBackendFailure reason:reason];
	}
	if (outHandler != NULL) {
		*outHandler = handler;
	}
	return YES;
}

+ (CGRect)contractRect:(CGRect)rect
{
	return CGRectMake(rect.origin.x, 1.0 - (rect.origin.y + rect.size.height), rect.size.width, rect.size.height);
}

+ (CGPoint)contractPoint:(CGPoint)point
{
	return CGPointMake(point.x, 1.0 - point.y);
}

+ (BOOL)failWithError:(NSError **)error code:(NSInteger)code reason:(NSString *)reason
{
	if (error != NULL) {
		*error = [NSError errorWithDomain:NFKInferenceErrorDomain
									 code:code
								 userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return NO;
}

@end
