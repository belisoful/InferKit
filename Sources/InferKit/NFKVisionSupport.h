//
//  NFKVisionSupport.h
//  InferKit
//
//  Private: the plumbing every Vision-framework backend repeats. Reading the request's image,
//  running requests against it, and turning Vision's geometry into the contract's.
//

#ifndef NFKVisionSupport_h
#define NFKVisionSupport_h

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <Vision/Vision.h>

@class NFKInferenceRequest;

NS_ASSUME_NONNULL_BEGIN

@interface NFKVisionSupport : NSObject

- (instancetype)init NS_UNAVAILABLE;

/*! The request's image for a key, as a CGImage, or NULL with kNFKError_InferenceMissingInput. */
+ (nullable CGImageRef)imageForRequest:(NFKInferenceRequest *)request
								   key:(NSString *)key
								 error:(NSError **)error CF_RETURNS_RETAINED;

/*! Runs the Vision requests against the image, reporting a failure as kNFKError_InferenceBackendFailure.
	The handler is passed back because an instance-mask observation needs it to produce its mask. */
+ (BOOL)performRequests:(NSArray<VNRequest *> *)requests
				onImage:(CGImageRef)image
				handler:(VNImageRequestHandler * _Nullable * _Nullable)outHandler
				  error:(NSError **)error;

/*! Vision normalizes with the origin at the lower left; the contract puts it at the top left. */
+ (CGRect)contractRect:(CGRect)rect;

/*! The point counterpart of contractRect:. */
+ (CGPoint)contractPoint:(CGPoint)point;

/*! Fails with a code from NFKErrors.h and a reason, and returns NO so a caller can `return` it. */
+ (BOOL)failWithError:(NSError **)error code:(NSInteger)code reason:(NSString *)reason;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKVisionSupport_h */
