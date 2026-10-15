//
//  NFKModelOptions.h
//  InferKit
//

#ifndef NFKModelOptions_h
#define NFKModelOptions_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKModelDescriptor.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKModelCatalog;

/*!
	@header     NFKModelOptions
	@abstract   Checking a set of option values against a descriptor, and resolving it into the values a
				build or a request uses.
	@discussion Option values arrive from a user interface, a configuration file, or a request body, so
				a value can be a string where the option takes a number. Resolving coerces such a value,
				fills each absent option's default, and reports every value it cannot use as an
				NFKModelOptionProblem. Introduced in InferKit 0.4.0.
*/

#pragma mark Problem codes

/*! The descriptor has no option under the key. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionProblemUnknownKey;
/*! The value is not of the option's kind and does not convert to it. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionProblemWrongType;
/*! The value is below the option's minimum. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionProblemBelowMinimum;
/*! The value is above the option's maximum. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionProblemAboveMaximum;
/*! The value is not a whole number of steps from the option's minimum. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionProblemOffStep;
/*! The value is none of the option's choices. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionProblemNotAChoice;
/*! This machine does not meet the requirements of the option or of the chosen choice. Introduced in
	InferKit 0.4.0. */
extern NSString * const NFKModelOptionProblemUnavailable;
/*! The option applies at another stage. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionProblemWrongStage;
/*! A model-reference option names a model the catalog does not have. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionProblemUnknownModel;
/*! A file-or-folder option names a path that does not exist. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionProblemMissingFile;

/*! The userInfo key of an NFKModelCatalogErrorInvalidOptions error: the NSArray of
	NFKModelOptionProblem. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionProblemsKey;

/*!
	@class      NFKModelOptionProblem
	@abstract   One option value a descriptor cannot use, and why.
	@discussion code is an NFKModelOptionProblem* string, and message a sentence a user interface shows
				beside the control. Introduced in InferKit 0.4.0.
*/
@interface NFKModelOptionProblem : NSObject <NSCopying>

/*! The option's key. */
@property (nonatomic, readonly, copy) NSString *key;
/*! What is wrong, an NFKModelOptionProblem* string. */
@property (nonatomic, readonly, copy) NSString *code;
/*! The sentence that explains the problem. */
@property (nonatomic, readonly, copy) NSString *message;
/*! The value as it was given. */
@property (nonatomic, readonly, copy, nullable) id value;

/*! A problem with every field. */
- (instancetype)initWithKey:(NSString *)key
					   code:(NSString *)code
					message:(NSString *)message
					  value:(nullable id)value NS_DESIGNATED_INITIALIZER;

/*! The problem as a dictionary: key, code, message, and value when it is a JSON value. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@category   NFKModelDescriptor (NFKModelOptionValidation)
	@abstract   Checks and resolves option values against the descriptor's options.
	@discussion stage is NFKModelOptionStageLoad, NFKModelOptionStageRequest, or nil for both. A value of
				the other stage is a wrong-stage problem, with two exceptions:
				- at the request stage, the options the memory model names as the context length, the
				  cache bits, and the cache group size, which a request may override;
				- NFKModelLoadOptionIgnoreMemoryBudget, which every load accepts.
				Values convert as follows:
				- integer: a whole NSNumber, or a string of one such as "8192".
				- number: an NSNumber, or a numeric string.
				- boolean: an NSNumber, or "true", "false", "yes", "no", "1", or "0".
				- choice: a value equal to a choice's, or whose text equals a choice's text; the result
				  is the choice's own value.
				- text: an NSString. text-list: an array of strings, or one string.
				- file-or-folder: a path string or a file NSURL, as a path; the path has to exist.
				- model-reference: the identifier of a descriptor in NFKModelCatalog.sharedCatalog. For
				  the memory model's draft option the descriptor is one of the option's choices when it
				  lists some, else a compatible draft (draftDescriptorsForVariant:inCatalog:).
				- json: any JSON value.
				Introduced in InferKit 0.4.0.
*/
@interface NFKModelDescriptor (NFKModelOptionValidation)

/*! Every problem with the values, in the order of their keys. Empty when every value resolves. */
- (NSArray<NFKModelOptionProblem *> *)problemsWithOptions:(nullable NSDictionary<NSString *, id> *)options
													stage:(nullable NSString *)stage;

/*! The values converted to each option's kind, with each absent option of the stage at its default.
	Returns nil and sets error (NFKModelCatalogErrorInvalidOptions, the problems under
	NFKModelOptionProblemsKey) when a value does not resolve. NFKModelLoadOptionIgnoreMemoryBudget is
	not in the result. */
- (nullable NSDictionary<NSString *, id> *)resolvedOptions:(nullable NSDictionary<NSString *, id> *)options
													 stage:(nullable NSString *)stage
													 error:(NSError **)error;

/*! A string that is equal for two sets of load options exactly when they build the same backend: the
	resolved load options, defaults filled, as JSON with sorted keys. nil when the options do not
	resolve. */
- (nullable NSString *)canonicalLoadOptionsKey:(nullable NSDictionary<NSString *, id> *)loadOptions;

/*! The descriptors in a catalog that can draft tokens for a variant of this model: the same family and
	vocabulary size and fewer parameters, in the catalog's order. A nil variant is the default variant.
	The choices of the memory model's draft option replace this list when there are some. */
- (NSArray<NFKModelDescriptor *> *)draftDescriptorsForVariant:(nullable NFKModelVariant *)variant
													inCatalog:(NFKModelCatalog *)catalog;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKModelOptions_h */
