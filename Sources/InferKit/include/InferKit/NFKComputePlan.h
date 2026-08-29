//
//  NFKComputePlan.h
//  InferKit
//

#ifndef NFKComputePlan_h
#define NFKComputePlan_h

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKComputePlan
	@abstract   Where Core ML plans to run a model's operations: the Neural Engine, the GPU, or the
				CPU.
	@discussion MLComputeUnits is a request. Core ML places each operation on the fastest device that
				supports it and reports nothing about the ones it moved, so a model asked for the
				Neural Engine can run entirely on the CPU and behave exactly as if it had not — only
				slower and warmer. That silence is what this answers.

				A plan is a per-operation count, so a partial placement is visible as one: a language
				model whose attention falls back to the GPU shows as a mixture rather than a yes or a
				no. neuralEngineFraction is the single number to watch, and describedPlacement is the
				line to log.

				This reads a COMPILED model (.mlmodelc) without running it, so it costs no inference
				and needs no privileges. It is a diagnostic: measure once while tuning a conversion,
				not on every load.

				Requires macOS 14.4 / iOS 17.4 / tvOS 17.4 — Core ML publishes no placement
				information before that. isAvailable reports whether the OS can answer at all, and
				planForCompiledModelAtURL:… fails with kNFKError_InferenceUnsupported on an
				older system rather than guessing.

				Introduced in InferKit 0.1.0.
*/
@interface NFKComputePlan : NSObject

/*! Whether this OS publishes compute-plan information (macOS 14.4 / iOS 17.4 / tvOS 17.4). */
@property (class, nonatomic, readonly) BOOL isAvailable;

/*!
	@method     planForCompiledModelAtURL:computeUnits:error:
	@abstract   Loads the plan for the compiled model at url, as it would run under computeUnits.
	@discussion url must be an .mlmodelc directory — the same one a backend loads. Pass the compute
				units the backend will use, since the placement depends on them.

				Blocks until Core ML answers; call it off the main thread.
	@result     The plan, or nil with an error.
*/
+ (nullable instancetype)planForCompiledModelAtURL:(NSURL *)url
									  computeUnits:(MLComputeUnits)computeUnits
											 error:(NSError * _Nullable *)outError;

/*!
	@method     planForCompiledModelAtURL:computeUnits:completionHandler:
	@abstract   The same, without blocking: the handler receives the plan or an error.
	@discussion Core ML's own compute-plan API is asynchronous, so this is the path that does no
				waiting. The handler runs on whichever queue Core ML answers on. The blocking variant
				above is the semaphore-wrapped form of this one, for the synchronous contract the rest
				of the toolkit uses.
*/
+ (void)planForCompiledModelAtURL:(NSURL *)url
					 computeUnits:(MLComputeUnits)computeUnits
				completionHandler:(void (^)(NFKComputePlan * _Nullable plan,
											NSError * _Nullable error))completionHandler;

/*! Operations Core ML prefers to run on the Neural Engine. */
@property (nonatomic, readonly) NSInteger neuralEngineOperationCount;

/*! Operations Core ML prefers to run on the GPU. */
@property (nonatomic, readonly) NSInteger gpuOperationCount;

/*! Operations Core ML prefers to run on the CPU. */
@property (nonatomic, readonly) NSInteger cpuOperationCount;

/*! Operations whose placement Core ML would not report. */
@property (nonatomic, readonly) NSInteger unknownOperationCount;

/*! Every operation the plan covers. */
@property (nonatomic, readonly) NSInteger operationCount;

/*! The share of operations on the Neural Engine, 0 to 1, or 0 when there are none to place. */
@property (nonatomic, readonly) double neuralEngineFraction;

/*! Whether every placeable operation is on the Neural Engine. */
@property (nonatomic, readonly) BOOL runsEntirelyOnNeuralEngine;

/*!
	@property   operatorNamesOffNeuralEngine
	@abstract   The distinct operator names Core ML placed somewhere other than the Neural Engine,
				most frequent first.
	@discussion This is the list to work from when a conversion is being tuned: one unsupported
				operator in the middle of a network splits it and costs more than its own share of
				the time, because the intermediate results cross devices.
*/
@property (nonatomic, readonly) NSArray<NSString *> *operatorNamesOffNeuralEngine;

/*! A one-line summary, for a log: counts per device and the Neural Engine share. */
@property (nonatomic, readonly) NSString *describedPlacement;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKComputePlan_h */
