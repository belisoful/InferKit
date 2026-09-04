//
//  NFKInferenceResult.h
//  InferKit
//

#ifndef NFKInferenceResult_h
#define NFKInferenceResult_h

#import <Foundation/Foundation.h>

@class NFKDetection;
@class NFKKeypoint;
@class NFKClassification;
@class NFKAudioSegment;

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKInferenceResult
	@abstract   The immutable output of one inference run: the model's named outputs.
	@discussion Mirrors NFKInferenceRequest. Outputs are keyed
				by the model's output names; values are opaque to the result, the same media
				and tensor types a request carries. A backend returns one result, or nil with
				an NSError. An image effect reads the output it expects by name and composites
				it; a passthrough backend returns the inputs it was given, so an effect renders
				its source unchanged with no model present.
*/
@interface NFKInferenceResult : NSObject <NSCopying>

/*! The model's named outputs. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *outputs;

/*! Convenience for NFKOutputText: the generated text, or nil when absent or not a string. Image,
	mask, and video outputs stay on outputForKey: because their value type is chosen by the backend
	(a CVPixelBuffer, a texture, a CGImage). */
@property (nonatomic, readonly, nullable) NSString *text;

/*! Convenience for NFKOutputStructured: the structured result keyed by field name, or nil. */
@property (nonatomic, readonly, nullable) NSDictionary<NSString *, id> *structured;

/*! NFKOutputToolCalls as an array of {id, name, arguments} dictionaries, or nil when absent or of another type. Introduced in InferKit 0.3.0. */
@property (nonatomic, readonly, nullable) NSArray<NSDictionary<NSString *, id> *> *toolCalls;

/*! Convenience for NFKOutputEmbedding: the feature embedding vector, or nil when absent or not an
	array of numbers. */
@property (nonatomic, readonly, nullable) NSArray<NSNumber *> *embedding;

/*! Convenience for NFKOutputDetections: the detected objects, or nil when absent or not an array. */
@property (nonatomic, readonly, nullable) NSArray<NFKDetection *> *detections;

/*! Convenience for NFKOutputPose: the located landmarks, or nil when absent or not an array. */
@property (nonatomic, readonly, nullable) NSArray<NFKKeypoint *> *pose;

/*! Convenience for NFKOutputClassifications: the predicted classes, or nil when absent or not an array. */
@property (nonatomic, readonly, nullable) NSArray<NFKClassification *> *classifications;

/*! Convenience for NFKOutputSegments: the located time spans, or nil when absent or not an array. */
@property (nonatomic, readonly, nullable) NSArray<NFKAudioSegment *> *segments;

+ (instancetype)resultWithOutputs:(NSDictionary<NSString *, id> *)outputs;

- (instancetype)initWithOutputs:(NSDictionary<NSString *, id> *)outputs NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

/*! The output for a key, or nil when absent. */
- (nullable id)outputForKey:(NSString *)key;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKInferenceResult_h */
