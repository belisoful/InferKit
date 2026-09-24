//
//  NFKDecisionAnswer.h
//  InferKit
//

#ifndef NFKDecisionAnswer_h
#define NFKDecisionAnswer_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKDecisionQuestion.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKDecisionAnswer
	@abstract   A decision model's typed answer to one NFKDecisionQuestion.
	@discussion A decision backend returns a map of these under NFKOutputAnswers, keyed as the
				questions were. The fields that carry the answer follow the type:

				- NFKDecisionTypeChoice → choice names the option picked, probabilities carries
				  every option's probability, and confidence how sure the pick is
				- NFKDecisionTypeScore → score is the expected level, a non-integer between the
				  levels when the model is torn, probabilities is keyed by level index, legend maps
				  each index to its description, and confidence how sure the level is
				- NFKDecisionTypeNoul → probability is the probability the statement holds

				A field the type does not carry reads as nil or zero. raw keeps the answer as the
				service returned it, for a field this type does not name. The type archives with
				secure coding on, like every other result value type. Introduced in InferKit 0.4.0.
*/
@interface NFKDecisionAnswer : NSObject <NSCopying, NSSecureCoding>

/*! The type of the question this answers. */
@property (nonatomic, readonly) NFKDecisionType type;

/*! A choice's picked option name. nil for the other types. */
@property (nonatomic, readonly, nullable, copy) NSString *choice;

/*! A score's expected level, 0 for the lowest; fractional between two levels. 0 for the other types. */
@property (nonatomic, readonly) double score;

/*! A noul's probability that the statement holds, in 0...1. 0 for the other types. */
@property (nonatomic, readonly) double probability;

/*! How sure the model is of a choice or a score, in 0...1; 0 where the service reports none. */
@property (nonatomic, readonly) double confidence;

/*! A choice's probability per option name, or a score's per level index written as a string. nil for
	a noul. */
@property (nonatomic, readonly, nullable, copy) NSDictionary<NSString *, NSNumber *> *probabilities;

/*! A score's level description per level index written as a string. nil for the other types. */
@property (nonatomic, readonly, nullable, copy) NSDictionary<NSString *, NSString *> *legend;

/*! The answer as the service returned it. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *raw;

/*!
	@method     answerWithDictionary:
	@abstract   Reads an answer from the wire shape a decision service returns.
	@discussion The dictionary's type names the fields read: choice, score, or noul beside
				probabilities, legend, and confidence. Returns nil when the type is absent or not one
				of the three.
*/
+ (nullable instancetype)answerWithDictionary:(NSDictionary<NSString *, id> *)dictionary;

- (instancetype)initWithType:(NFKDecisionType)type
					  choice:(nullable NSString *)choice
					   score:(double)score
				 probability:(double)probability
				  confidence:(double)confidence
			   probabilities:(nullable NSDictionary<NSString *, NSNumber *> *)probabilities
					  legend:(nullable NSDictionary<NSString *, NSString *> *)legend
						 raw:(nullable NSDictionary<NSString *, id> *)raw NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKDecisionAnswer_h */
