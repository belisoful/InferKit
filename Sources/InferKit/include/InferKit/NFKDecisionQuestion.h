//
//  NFKDecisionQuestion.h
//  InferKit
//

#ifndef NFKDecisionQuestion_h
#define NFKDecisionQuestion_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@enum       NFKDecisionType
	@abstract   The three shapes a typed decision takes.
	@discussion A decision model answers a question with a value of a fixed type rather than with
				text: one option out of a named set, a level on an ordered scale, or the
				probability that a statement holds. Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKDecisionType) {
	/*! One option out of a named set: classification, routing, selection. */
	NFKDecisionTypeChoice = 0,
	/*! A level on an ordered scale: severity, priority, quality. */
	NFKDecisionTypeScore = 1,
	/*! Whether a statement holds, as a probability: a yes-or-no judgment. */
	NFKDecisionTypeNoul = 2,
};

/*!
	@class      NFKDecisionQuestion
	@abstract   One typed question a decision model answers about a state.
	@discussion A request to a decision backend carries a state under NFKInputState and a map of
				these under NFKInputQuestions, keyed by the caller's own identifiers; each is
				answered by an NFKDecisionAnswer of the same type under the same key. The three
				factories build the three types:

				- a choice names its options in order, each with an optional description
				- a score names its levels in order, from lowest to highest, two to ten of them
				- a noul carries the statement in its instructions, and optionally what "true" and
				  "false" mean for it

				dictionaryRepresentation is the wire shape the hosted and on-device decision
				backends read ({type, instructions, criteria}), so a caller that already holds a
				question in that shape passes the dictionary under NFKInputQuestions directly.
				Introduced in InferKit 0.4.0.
*/
@interface NFKDecisionQuestion : NSObject <NSCopying>

/*! What kind of answer the question takes. */
@property (nonatomic, readonly) NFKDecisionType type;

/*! What is being asked, in prose. */
@property (nonatomic, readonly, copy) NSString *instructions;

/*! A choice's option names in order, or a score's level descriptions from lowest to highest. Empty
	for a noul. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *options;

/*! A choice's option descriptions keyed by option name, a noul's meanings keyed by "true" and
	"false"; only the ones the caller gave. Empty for a score, whose levels are their descriptions. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, NSString *> *descriptions;

/*! A choice among named options. */
+ (instancetype)choiceQuestionWithInstructions:(NSString *)instructions
									   options:(NSArray<NSString *> *)options;

/*! A choice among named options, each described. An option absent from descriptions is sent with
	no description. */
+ (instancetype)choiceQuestionWithInstructions:(NSString *)instructions
									   options:(NSArray<NSString *> *)options
								  descriptions:(nullable NSDictionary<NSString *, NSString *> *)descriptions;

/*! A score on an ordered scale. levels describes each level from lowest to highest. */
+ (instancetype)scoreQuestionWithInstructions:(NSString *)instructions
									   levels:(NSArray<NSString *> *)levels;

/*! A yes-or-no judgment of the statement in instructions. */
+ (instancetype)noulQuestionWithInstructions:(NSString *)instructions;

/*! A yes-or-no judgment with what each answer means spelled out. */
+ (instancetype)noulQuestionWithInstructions:(NSString *)instructions
								trueMeaning:(nullable NSString *)trueMeaning
							   falseMeaning:(nullable NSString *)falseMeaning;

- (instancetype)initWithType:(NFKDecisionType)type
				instructions:(NSString *)instructions
					 options:(nullable NSArray<NSString *> *)options
				descriptions:(nullable NSDictionary<NSString *, NSString *> *)descriptions NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

/*!
	@method     questionWithDictionary:
	@abstract   Reads a question from the wire shape a decision service takes.
	@discussion The inverse of dictionaryRepresentation: type names the type, instructions the prose,
				and criteria the options (a choice's map of name to description or null, read in
				sorted name order because a dictionary carries none; a score's ordered levels; a
				noul's "true" / "false" meanings). Returns nil when the type is absent or not one of
				the three, or the instructions are not a string.
*/
+ (nullable instancetype)questionWithDictionary:(NSDictionary<NSString *, id> *)dictionary;

/*!
	@method     dictionaryRepresentation
	@abstract   The question in the wire shape a decision service reads.
	@discussion type is "choice", "score", or "noul"; instructions is the prose; criteria is a map
				of option name to description or null for a choice, the ordered level descriptions
				for a score, and the "true" / "false" meanings for a noul, present only when the
				caller gave one.
*/
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

/*! The wire name of a type: "choice", "score", or "noul". */
+ (NSString *)nameForType:(NFKDecisionType)type;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKDecisionQuestion_h */
