//
//  NFKLiveStream.h
//  InferKit
//

#ifndef NFKLiveStream_h
#define NFKLiveStream_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@protocol   NFKLiveStream
	@abstract   A model run on a live source, one unit at a time, with the state it carries between units.
	@discussion A unit is a video frame or an audio chunk, as the model's live profile declares
				(NFKModelLive.h). Each unit is one request whose inputs carry the unit under the profile's
				input keys; its result carries the profile's output keys. The parameters the stream was
				opened with apply to every unit, and a unit's own parameters override them.

				A stateful model keeps its state from one unit to the next, so a stream takes its units in
				order: a recurrent matter carries its hidden state, and a speech enhancer its overlap and
				lookahead. A stream answers units one at a time; a caller that pushes from two threads
				gets them answered one after the other. Introduced in InferKit 0.4.0.
*/
@protocol NFKLiveStream <NSObject>

/*! Runs one unit and returns its result, or nil with an error. A stateful stream answers it from the
	state the units before it left. */
- (nullable NFKInferenceResult *)resultForUnit:(NFKInferenceRequest *)unit error:(NSError **)error
	NS_SWIFT_NAME(result(for:));

/*! Forgets the carried state, so the next unit runs as the first of a new stream. */
- (void)reset;

/*! Releases what the stream holds. A unit pushed after close fails with kNFKError_InferenceNotReady. */
- (void)close;

@end

/*! Answers one unit, its parameters already merged over the stream's. Returns nil and fills error on a
	failure. */
typedef NFKInferenceResult * _Nullable (^NFKLiveStreamUnitHandler)(NFKInferenceRequest *unit, NSError **error);

/*!
	@class      NFKBlockLiveStream
	@abstract   A live stream whose units, reset, and close are blocks.
	@discussion The stream merges its parameters under each unit's own, answers one unit at a time, and
				refuses units after close with kNFKError_InferenceNotReady in NFKInferenceErrorDomain. A
				backend keeps its carried state where the blocks reach it, and returns one of these from
				openLiveStreamWithParameters:error:. reset and close run their blocks between units, never
				during one. Introduced in InferKit 0.4.0.
*/
@interface NFKBlockLiveStream : NSObject <NFKLiveStream>

/*! The parameters every unit runs with, under the unit's own. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *parameters;

/*! YES once close has run. */
@property (nonatomic, readonly, getter=isClosed) BOOL closed;

/*! A stream that answers units with unitHandler, forgets state with resetHandler, and releases with
	closeHandler. */
- (instancetype)initWithParameters:(nullable NSDictionary<NSString *, id> *)parameters
					   unitHandler:(NFKLiveStreamUnitHandler)unitHandler
					  resetHandler:(nullable void (^)(void))resetHandler
					  closeHandler:(nullable void (^)(void))closeHandler NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@function   NFKInferenceOpenLiveStream
	@abstract   Opens a live stream on a backend, uniformly across backends that carry state and backends
				that do not.
	@discussion When the backend implements openLiveStreamWithParameters:error:, this calls it and returns
				its stream. Otherwise it returns a stream that runs each unit through
				runInferenceForRequest:error: with parameters merged under the unit's own, which serves a
				model that treats each unit alone. Introduced in InferKit 0.4.0.
*/
id<NFKLiveStream> _Nullable NFKInferenceOpenLiveStream(id<NFKInferenceBackend> backend,
													   NSDictionary<NSString *, id> * _Nullable parameters,
													   NSError * _Nullable * _Nullable error);

NS_ASSUME_NONNULL_END

#endif /* NFKLiveStream_h */
