//
//  InferKit.h
//  InferKit
//
//  A cross-platform inference toolkit: a swappable backend protocol, request/result value
//  types, the shipped passthrough / Core ML / remote backends, the texture-tensor conversion,
//  and the Hugging Face access layer. No FxPlug or host-framework dependency.
//

#ifndef InferKit_h
#define InferKit_h

#import <InferKit/NFKInferKit.h>
#import <InferKit/NFKErrors.h>
#import <InferKit/NFKModality.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKVideoAsset.h>
#import <InferKit/NFKAudioAsset.h>
#import <InferKit/NFKDetection.h>
#import <InferKit/NFKKeypoint.h>
#import <InferKit/NFKClassification.h>
#import <InferKit/NFKAudioSegment.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceBackend.h>
#import <InferKit/NFKDynamicBackend.h>
#import <InferKit/NFKPassthroughBackend.h>
#import <InferKit/NFKCoreMLBackend.h>
#import <InferKit/NFKCoreMLLanguageBackend.h>
#import <InferKit/NFKComputePlan.h>
#import <InferKit/NFKHardwareProfile.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKAnthropicBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTranscriptionBackend.h>
#import <InferKit/NFKAsyncGenerationBackend.h>
#import <InferKit/NFKTensorConversion.h>
#import <InferKit/NFKMLMultiArray.h>
#import <InferKit/NFKTokenizer.h>
#import <InferKit/NFKHFHub.h>

#endif /* InferKit_h */
