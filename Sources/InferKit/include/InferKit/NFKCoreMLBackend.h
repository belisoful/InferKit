//
//  NFKCoreMLBackend.h
//  InferKit
//

#ifndef NFKCoreMLBackend_h
#define NFKCoreMLBackend_h

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKCoreMLBackend
	@abstract   The default in-process engine: it runs a Core ML model on the GPU or Neural
				Engine.
	@discussion Depends only on Apple frameworks, so InferKit ships it.
				It owns the bridge between NFK's image representation and Core ML's features:
				an id<MTLTexture> input becomes a CVPixelBuffer over the texture's IOSurface, a
				CVPixelBuffer or MLMultiArray passes straight through, and a scalar becomes the
				matching MLFeatureValue. A model's image output returns as an id<MTLTexture> the
				effect template can composite; other outputs return as their Core ML objects for
				a subclass to handle.

				The model is set by URL. prepareWithError: loads it, compiling an .mlpackage or
				.mlmodel to an .mlmodelc first when needed. Loading is slow, so a caller prepares
				once off the render thread. isReady reports whether a model is loaded.
*/
@interface NFKCoreMLBackend : NSObject <NFKInferenceBackend>

/*! The model to load: an .mlmodelc, .mlpackage, or .mlmodel URL. */
@property (nonatomic, copy, nullable) NSURL *modelURL;

/*!
	@property   computeUnits
	@abstract   The Core ML compute units the model loads with. Defaults to MLComputeUnitsAll.
	@discussion Set it before loadModelFromURL:error:. MLComputeUnitsAll lets Core ML place each
				operation, which is usually what a caller wants. The value is a request, not a
				guarantee: Core ML places an operation the Neural Engine cannot run elsewhere, and
				says nothing about having done so. NFKComputePlan reports where the operations
				actually land.
*/
@property (nonatomic, assign) MLComputeUnits computeUnits;

+ (instancetype)backendWithModelURL:(nullable NSURL *)modelURL;

/*!
	@method     loadModelFromURL:error:
	@abstract   Loads the model at url now, compiling it first when needed; YES on success.
	@discussion Sets modelURL and leaves the backend ready on success. Returns NO with an error
				when the URL is missing, the compile fails, or the model cannot be loaded.
*/
- (BOOL)loadModelFromURL:(NSURL *)url error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKCoreMLBackend_h */
