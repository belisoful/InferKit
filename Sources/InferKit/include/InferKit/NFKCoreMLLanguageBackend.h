//
//  NFKCoreMLLanguageBackend.h
//  InferKit
//

#ifndef NFKCoreMLLanguageBackend_h
#define NFKCoreMLLanguageBackend_h

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKCoreMLLanguageBackend
	@abstract   Runs a converted causal language model on device through Core ML.
	@discussion Depends only on Apple frameworks, so InferKit ships it.
				Core ML runs one forward pass; it does not tokenize, sample, or loop. This backend
				owns that work: it tokenizes the prompt, runs the model with a Core ML state that
				holds the KV cache, samples a token from the logits, feeds it back, and repeats until
				a stop condition. It returns the generated text under NFKOutputText.

				The model is a directory the inferkit-convert tool produces: a stateful model.mlpackage,
				the tokenizer files, and a manifest.json naming the model's input and logits features
				and the tokenizer type. loadModelFromDirectory: reads the manifest, compiles and loads
				the model, and builds the tokenizer. Loading is slow, so a caller prepares once off
				the render thread.

				A request supplies its prompt through NFKInputPrompt (a string) or NFKInputMessages
				(an OpenAI-style array). Generation reads the text parameters (NFKParameterTemperature,
				NFKParameterTopP, NFKParameterTopK, NFKParameterMaxTokens, NFKParameterRepetitionPenalty,
				NFKParameterStopSequences, NFKParameterSeed). submitInferenceJobForRequest: reports
				per-token progress and honors cancellation.

				Core ML state requires macOS 15 / iOS 18, so the class is annotated accordingly. It
				does not raise the core's deployment floor: it is a runtime-gated symbol.
*/
API_AVAILABLE(macos(15.0), ios(18.0), tvos(18.0))
@interface NFKCoreMLLanguageBackend : NSObject <NFKInferenceBackend>

/*! The model directory: a folder holding model.mlpackage (or model.mlmodelc), manifest.json, and the
	tokenizer files. */
@property (nonatomic, copy, nullable) NSURL *modelDirectoryURL;

/*!
	@property   computeUnits
	@abstract   The Core ML compute units the model loads with. Defaults to MLComputeUnitsAll.
	@discussion Set it before prepareWithError: or loadModelFromDirectory:error:. MLComputeUnitsAll
				lets Core ML place each op on CPU, GPU, or the Neural Engine and is the fastest for a
				converted language model. MLComputeUnitsCPUAndNeuralEngine can fail to load a stateful
				model: the KV-cache scatter does not compile for the Neural Engine. Use All (the
				default) or CPUOnly.
*/
@property (nonatomic, assign) MLComputeUnits computeUnits;

+ (instancetype)backendWithModelDirectoryURL:(nullable NSURL *)modelDirectoryURL;

/*!
	@method     loadModelFromDirectory:error:
	@abstract   Loads the manifest, model, and tokenizer from directoryURL now; YES on success.
	@discussion Sets modelDirectoryURL and leaves the backend ready on success. Returns NO with an
				error when the manifest is missing or malformed, the tokenizer type is unsupported,
				or the model cannot be compiled or loaded.
*/
- (BOOL)loadModelFromDirectory:(NSURL *)directoryURL error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKCoreMLLanguageBackend_h */
