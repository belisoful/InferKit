//
//  NFKLocalModelRunner.h
//  InferKit
//

#ifndef NFKLocalModelRunner_h
#define NFKLocalModelRunner_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteModel.h>
#import <InferKit/NFKInferenceJob.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@protocol   NFKLocalModelRunner
	@abstract   A model runner on this machine, managed through its own API.
	@discussion The OpenAI-compatible surface every runner serves answers what a model does with a
				prompt; it says nothing about what is installed, what is loaded, how large a model
				is, or how to get one. Each runner answers those through a native API of its own,
				and this protocol is the shape they share so an app treats "a runner on this
				machine" uniformly: NFKOllamaRunner and NFKLMStudioRunner adopt it, and
				-[NFKRemoteProvider localRunner] hands back the adapter for a preset.

				The required methods read; the optional ones change the machine (a pull downloads
				weights, a delete removes them), so an adapter offers them only where its runner
				does, and a caller checks respondsToSelector: before offering the action. Every
				call blocks and is run off the render thread; a pull returns a job instead, for
				progress and cancellation. A runner that is not running fails with
				kNFKError_RemoteUnreachable. Introduced in InferKit 0.3.0.
*/
@protocol NFKLocalModelRunner <NSObject>

/*! The preset this runner manages. Its baseURL is where the OpenAI-compatible surface is. */
@property (nonatomic, copy, readonly) NFKRemoteProvider *provider;

/*! Where the runner's native API is: the provider's base with its /v1 removed. */
@property (nonatomic, copy, readonly) NSURL *nativeBaseURL;

/*! Whether the runner answers at all. A rejected request still counts; a refused connection does not. */
- (BOOL)isRunning;

/*! The models installed on this machine, with the size, quantization, context length, and
	capabilities the runner reports for each. */
- (nullable NSArray<NFKRemoteModel *> *)installedModelsWithError:(NSError * _Nullable *)outError;

/*! The models currently loaded in memory, a subset of the installed ones. */
- (nullable NSArray<NFKRemoteModel *> *)loadedModelsWithError:(NSError * _Nullable *)outError;

/*! One installed model in detail, or nil with an error naming the missing model. */
- (nullable NFKRemoteModel *)detailsForModel:(NSString *)identifier error:(NSError * _Nullable *)outError;

@optional

/*! The runner's version string, where it publishes one. */
- (nullable NSString *)versionWithError:(NSError * _Nullable *)outError;

/*!
	@method     pullModel:
	@abstract   Downloads a model into the runner, reporting progress through the job.
	@discussion The job's progress is the fraction of the layer being fetched; its partialResult
				carries the runner's latest status line under NFKOutputText. It finishes with the
				identifier under NFKOutputText, or with kNFKError_InferenceBackendFailure carrying
				the runner's own message for a name it cannot find. Cancelling the job stops the
				download. This is the local counterpart of NFKHFHub.downloadRepo:.
*/
- (NFKInferenceJob *)pullModel:(NSString *)identifier;

/*! Removes a model's weights from this machine. Fails, naming the model, when it is not installed. */
- (BOOL)deleteModel:(NSString *)identifier error:(NSError * _Nullable *)outError;

@end

@interface NFKRemoteProvider (LocalRunner)

/*!
	@property   localRunner
	@abstract   The native-API adapter for this preset, or nil where there is none.
	@discussion The ollama and lmstudio presets have one. llama.cpp and vLLM serve nothing beyond
				the OpenAI-compatible surface that a runner adapter would add, so they answer nil
				and NFKRemoteModelCatalog is the whole of what can be asked of them. A re-based
				preset (providerWithBaseURL:) keeps its runner. A new adapter is built on each
				read. Introduced in InferKit 0.3.0.
*/
@property (nonatomic, readonly, nullable) id<NFKLocalModelRunner> localRunner;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKLocalModelRunner_h */
