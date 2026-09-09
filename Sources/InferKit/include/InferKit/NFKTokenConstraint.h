//
//  NFKTokenConstraint.h
//  InferKit
//
//  Grammar-constrained sampling for the core's own language backend: a mask over a logit buffer that
//  admits only the tokens a grammar can accept next.
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKTokenizer.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKTokenVocabulary
	@abstract   The bytes every token id decodes to, which is what a grammar reasons over.
	@discussion A grammar is defined over text, a model emits token ids, and a byte-level vocabulary
	is the bridge: a token is admissible when appending its bytes keeps the output inside the
	grammar. `size` is the model's logit width, which can exceed the tokenizer's vocabulary; an id
	with no bytes is never admitted. Introduced in InferKit 0.4.0.
*/
@interface NFKTokenVocabulary : NSObject

/*! Reads every id below `size` from the tokenizer. The end token defaults to the tokenizer's
	`eosTokenId` and is admitted only when the output is complete. */
- (instancetype)initWithTokenizer:(NFKTokenizer *)tokenizer size:(NSUInteger)size;

/*! Builds from explicit token bytes (one NSData per id; an empty NSData is an id with no bytes) and
	the id that ends the output, or -1 for none. */
- (instancetype)initWithTokens:(NSArray<NSData *> *)tokens endToken:(NSInteger)endToken NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

@property (nonatomic, readonly) NSUInteger count;
@property (nonatomic, readonly) NSInteger endToken;
/*! The bytes of one id; empty for an id the vocabulary does not hold. */
- (NSData *)bytesForToken:(NSUInteger)token;

@end

@class NFKTokenConstraintCursor;

/*!
	@class      NFKTokenConstraint
	@abstract   A constraint defined byte by byte, shared and immutable; `makeCursor` returns the
	per-run state that walks the output.
	@discussion The concrete grammars are `NFKJSONConstraint` (well-formed JSON) and
	`NFKChoiceConstraint` (exactly one of a fixed set). `NFKCoreMLLanguageBackend` builds one from the
	`NFKParameterOutputFormat` or `NFKParameterChoices` request parameter and applies its cursor's mask
	to the logits before sampling, so sampling and greedy decoding alike stay inside the grammar.
	Introduced in InferKit 0.4.0.
*/
@interface NFKTokenConstraint : NSObject

@property (nonatomic, readonly) NFKTokenVocabulary *vocabulary;

/*! Whether `text` is a prefix the grammar can still complete. */
- (BOOL)acceptsText:(NSString *)text;
/*! Whether `text` is a complete output. */
- (BOOL)isCompleteText:(NSString *)text;
/*! The ids admissible at the start of an output, in ascending order (for inspection and tests). */
- (NSArray<NSNumber *> *)allowedTokensAfterText:(NSString *)text;

- (NFKTokenConstraintCursor *)makeCursor;

@end

/*!
	@class      NFKTokenConstraintCursor
	@abstract   One run's position inside a constraint.
*/
@interface NFKTokenConstraintCursor : NSObject

/*! Sets every inadmissible score to `-INFINITY` in place. The end token is admitted only once the
	grammar is satisfied — and also when nothing else is admissible, so a run stops rather than emitting
	a token the grammar would refuse. */
- (void)maskScores:(float *)scores count:(NSUInteger)count;
/*! Records the token that was emitted. */
- (void)acceptToken:(NSInteger)token;
/*! Whether the output so far is a complete document. */
@property (nonatomic, readonly) BOOL isComplete;
/*! The id that ends the output, or -1. */
@property (nonatomic, readonly) NSInteger endToken;

@end

/*! What a JSON document's root may be. */
typedef NS_ENUM(NSInteger, NFKJSONRoot) {
	/*! An object or an array. */
	NFKJSONRootContainer = 0,
	/*! An object only. */
	NFKJSONRootObject,
	/*! An array only. */
	NFKJSONRootArray,
	/*! Any JSON value, scalars included. */
	NFKJSONRootAny,
};

/*!
	@class      NFKJSONConstraint
	@abstract   Constrains the output to well-formed JSON, terminated when the root value closes.
	@discussion This is syntax, not schema: the keys and types are the model's choice, the shape is
	guaranteed. Whitespace between tokens is admitted as JSON admits it but capped at
	`maximumWhitespaceRun` consecutive bytes (8 by default): a model whose preferred next token is
	forbidden takes the whitespace it is offered indefinitely otherwise. Strings take any UTF-8 and the
	standard escapes; a number is refused a leading zero or a trailing dot as the specification refuses
	them. Introduced in InferKit 0.4.0.
*/
@interface NFKJSONConstraint : NFKTokenConstraint

- (instancetype)initWithVocabulary:(NFKTokenVocabulary *)vocabulary root:(NFKJSONRoot)root;
- (instancetype)initWithVocabulary:(NFKTokenVocabulary *)vocabulary
							  root:(NFKJSONRoot)root
			  maximumWhitespaceRun:(NSUInteger)maximumWhitespaceRun NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

@property (nonatomic, readonly) NFKJSONRoot root;
@property (nonatomic, readonly) NSUInteger maximumWhitespaceRun;

@end

/*!
	@class      NFKChoiceConstraint
	@abstract   Constrains the output to exactly one of a fixed set of strings.
	@discussion A classification or a menu: the model picks among the choices and nothing else, ending
	as soon as a choice is spelled out in full. A choice that is a prefix of another ends only when the
	model emits the end token, which the grammar admits at any complete choice. Introduced in
	InferKit 0.4.0.
*/
@interface NFKChoiceConstraint : NFKTokenConstraint

- (instancetype)initWithChoices:(NSArray<NSString *> *)choices
					 vocabulary:(NFKTokenVocabulary *)vocabulary NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

@property (nonatomic, readonly, copy) NSArray<NSString *> *choices;

@end

NS_ASSUME_NONNULL_END
