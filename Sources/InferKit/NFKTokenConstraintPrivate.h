//
//  NFKTokenConstraintPrivate.h
//  InferKit
//
//  The byte-level machine the concrete grammars implement; not part of the public API.
//

#ifndef NFKTokenConstraintPrivate_h
#define NFKTokenConstraintPrivate_h

#import "NFKTokenConstraint.h"

NS_ASSUME_NONNULL_BEGIN

/*! The deepest nesting a grammar state tracks; a document nested further is refused. */
#define NFK_CONSTRAINT_MAX_DEPTH 64

/*! A grammar's position, as a plain value so a token's bytes can be tried on a copy. The layout is
	the JSON grammar's; the choice grammar uses `length` alone. */
typedef struct {
	uint8_t frames[NFK_CONSTRAINT_MAX_DEPTH];	// per level: kind and phase, packed
	uint8_t depth;
	uint8_t scalar;								// which scalar is in progress
	uint8_t phase;								// its phase (string phase, number phase, literal index)
	uint8_t literal;							// which literal, when scalar is a literal
	uint8_t inKey;
	uint8_t complete;
	uint8_t whitespaceRun;
	uint32_t length;							// bytes emitted (the choice grammar's whole state)
} NFKConstraintState;

@interface NFKTokenConstraint ()

- (instancetype)initWithVocabulary:(NFKTokenVocabulary *)vocabulary;

/*! The state before any output. */
- (NFKConstraintState)initialState;
/*! Advances `state` by one byte in place; returns NO when the grammar rejects it. */
- (BOOL)advanceState:(NFKConstraintState *)state byte:(uint8_t)byte;
/*! Whether the output may end in `state`. */
- (BOOL)isCompleteState:(const NFKConstraintState *)state;
/*! Advances by every byte; NO when any is rejected. */
- (BOOL)advanceState:(NFKConstraintState *)state bytes:(const uint8_t *)bytes length:(NSUInteger)length;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKTokenConstraintPrivate_h */
