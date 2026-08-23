//
//  NFKInferenceRequest.h
//  InferKit
//

#ifndef NFKInferenceRequest_h
#define NFKInferenceRequest_h

#import <Foundation/Foundation.h>
#import "NFKModality.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKInferenceRequest
	@abstract   The immutable input to one inference run: named inputs plus parameters.
	@discussion A request carries the model's named inputs and a
				separate bag of parameters, keeping the two roles distinct:

				- inputs are the tensors or media a backend consumes, keyed by the model's
				  input names. Values are opaque to the request: a CVPixelBuffer, an
				  MLMultiArray, an NFKImageBuffer, an NSString prompt, or NSData. The
				  backend interprets them.
				- parameters are scalar controls a backend reads, keyed by name: seed,
				  strength, step count, guidance scale, and similar. Values are typically
				  NSNumber or NSString.

				The split lets one request describe an image-to-image pass (inputs hold the
				plate and mask, parameters hold strength and seed) or a text pass (inputs
				hold the prompt) through the same type. The value is immutable; build a new
				request to change it.
*/
@interface NFKInferenceRequest : NSObject <NSCopying>

/*! The model's named inputs. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *inputs;

/*! The scalar controls a backend reads. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *parameters;

/*! The modality the caller wants back. Defaults to NFKModalityImage. Input modalities are implied
	by the input values, so text, image, and video inputs mix freely under one request. */
@property (nonatomic, readonly) NFKModality outputModality;

/*! Convenience for NFKInputPrompt: the prompt string, or nil when absent or not a string. Image,
	mask, and video inputs stay on inputForKey: because their value type is chosen by the caller. */
@property (nonatomic, readonly, nullable) NSString *prompt;

/*! Convenience for NFKInputNegativePrompt: the negative prompt string, or nil. */
@property (nonatomic, readonly, nullable) NSString *negativePrompt;

/*! Convenience for NFKInputMessages: an OpenAI-style array of {role, content} dictionaries, or nil. */
@property (nonatomic, readonly, nullable) NSArray<NSDictionary<NSString *, id> *> *messages;

+ (instancetype)requestWithInputs:(NSDictionary<NSString *, id> *)inputs
					   parameters:(nullable NSDictionary<NSString *, id> *)parameters
				   outputModality:(NFKModality)outputModality;

+ (instancetype)requestWithInputs:(NSDictionary<NSString *, id> *)inputs
					   parameters:(nullable NSDictionary<NSString *, id> *)parameters;

+ (instancetype)requestWithInputs:(NSDictionary<NSString *, id> *)inputs;

- (instancetype)initWithInputs:(NSDictionary<NSString *, id> *)inputs
					parameters:(nullable NSDictionary<NSString *, id> *)parameters
				outputModality:(NFKModality)outputModality NS_DESIGNATED_INITIALIZER;

- (instancetype)initWithInputs:(NSDictionary<NSString *, id> *)inputs
					parameters:(nullable NSDictionary<NSString *, id> *)parameters;

- (instancetype)init NS_UNAVAILABLE;

/*! The input for a key, or nil when absent. */
- (nullable id)inputForKey:(NSString *)key;

/*! The parameter for a key, or nil when absent. */
- (nullable id)parameterForKey:(NSString *)key;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKInferenceRequest_h */
