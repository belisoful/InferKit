//
//  NFKAppleProviders.h
//  InferKit
//

#ifndef NFKAppleProviders_h
#define NFKAppleProviders_h

#import <Foundation/Foundation.h>
#import "NFKDynamicBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@header     NFKAppleProviders
	@abstract   The dynamic-discovery providers for the core's Apple-framework engines.
	@discussion `NFKDynamicBackend` resolves a capability to the first provider class that is linked,
				and these are named last for their capabilities, behind the companion packages. A
				consumer that links InferKitMLX keeps getting the MLX model; a consumer that links
				nothing still gets an answer, because these ship in the core and need no download.

				A provider hands back nil where its engine cannot serve the machine, and discovery
				moves on to the next candidate. Introduced in InferKit 0.4.0.
*/

/*! Reads text with Vision. Answers NFKCapabilityTextRecognition. */
@interface NFKVisionTextProvider : NSObject <NFKDynamicBackendProvider>
@end

/*! Masks the subject with Vision. Answers NFKCapabilitySegmentation. */
@interface NFKVisionSegmentationProvider : NSObject <NFKDynamicBackendProvider>
@end

/*! Estimates body pose with Vision. Answers NFKCapabilityPose. */
@interface NFKVisionPoseProvider : NSObject <NFKDynamicBackendProvider>
@end

/*! Finds faces with Vision. Answers NFKCapabilityFaceDetection. */
@interface NFKVisionFaceProvider : NSObject <NFKDynamicBackendProvider>
@end

/*! Embeds an image with Vision. Answers NFKCapabilityImageEmbedding. */
@interface NFKVisionFeaturePrintProvider : NSObject <NFKDynamicBackendProvider>
@end

/*! Enlarges a frame with VideoToolbox, where the machine has the processor. Answers
	NFKCapabilityUpscaling. */
@interface NFKVideoToolboxUpscalingProvider : NSObject <NFKDynamicBackendProvider>
@end

/*! Estimates motion with VideoToolbox, where the machine has the processor. Answers
	NFKCapabilityOpticalFlow. */
@interface NFKVideoToolboxOpticalFlowProvider : NSObject <NFKDynamicBackendProvider>
@end

/*! Transcribes with Apple's recognizer. Answers NFKCapabilityTranscription behind
	NFKMLXWhisperProvider. */
@interface NFKSpeechRecognitionProvider : NSObject <NFKDynamicBackendProvider>
@end

NS_ASSUME_NONNULL_END

#endif /* NFKAppleProviders_h */
