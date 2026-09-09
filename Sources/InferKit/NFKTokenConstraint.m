//
//  NFKTokenConstraint.m
//  InferKit
//

#import "NFKTokenConstraint.h"
#import "NFKTokenConstraintPrivate.h"

#pragma mark - Vocabulary

@implementation NFKTokenVocabulary
{
	NSArray<NSData *> *_tokens;
}

- (instancetype)initWithTokenizer:(NFKTokenizer *)tokenizer size:(NSUInteger)size
{
	NSMutableArray<NSData *> *tokens = [NSMutableArray arrayWithCapacity:size];
	for (NSUInteger tokenId = 0; tokenId < size; tokenId++) {
		NSData *bytes = [tokenizer bytesForTokenId:(NSInteger)tokenId];
		[tokens addObject:bytes ?: [NSData data]];
	}
	return [self initWithTokens:tokens endToken:tokenizer.eosTokenId];
}

- (instancetype)initWithTokens:(NSArray<NSData *> *)tokens endToken:(NSInteger)endToken
{
	self = [super init];
	if (self != nil) {
		_tokens = [tokens copy];
		_endToken = endToken;
	}
	return self;
}

- (NSUInteger)count
{
	return _tokens.count;
}

- (NSData *)bytesForToken:(NSUInteger)token
{
	return token < _tokens.count ? _tokens[token] : [NSData data];
}

@end

#pragma mark - Cursor

@interface NFKTokenConstraintCursor ()
- (instancetype)initWithConstraint:(NFKTokenConstraint *)constraint;
@end

@implementation NFKTokenConstraintCursor
{
	NFKTokenConstraint *_constraint;
	NFKConstraintState _state;
}

- (instancetype)initWithConstraint:(NFKTokenConstraint *)constraint
{
	self = [super init];
	if (self != nil) {
		_constraint = constraint;
		_state = [constraint initialState];
	}
	return self;
}

- (NSInteger)endToken
{
	return _constraint.vocabulary.endToken;
}

- (BOOL)isComplete
{
	return [_constraint isCompleteState:&_state];
}

- (void)maskScores:(float *)scores count:(NSUInteger)count
{
	NFKTokenVocabulary *vocabulary = _constraint.vocabulary;
	NSInteger endToken = vocabulary.endToken;
	BOOL complete = [_constraint isCompleteState:&_state];
	BOOL anyAdmitted = NO;
	NSUInteger limit = MIN(count, vocabulary.count);
	for (NSUInteger token = 0; token < limit; token++) {
		if ((NSInteger)token == endToken) {
			if (!complete) {
				scores[token] = -INFINITY;
			}
			continue;
		}
		NSData *bytes = [vocabulary bytesForToken:token];
		NFKConstraintState trial = _state;
		if (bytes.length == 0 || ![_constraint advanceState:&trial bytes:bytes.bytes length:bytes.length]) {
			scores[token] = -INFINITY;
		} else {
			anyAdmitted = YES;
		}
	}
	for (NSUInteger token = limit; token < count; token++) {
		scores[token] = -INFINITY;
	}
	// No way forward at all: only the end can follow, so the run stops rather than emitting a token
	// the grammar would refuse.
	if (!anyAdmitted && endToken >= 0 && (NSUInteger)endToken < count) {
		scores[endToken] = 0.0f;
	}
}

- (void)acceptToken:(NSInteger)token
{
	if (token < 0 || token == _constraint.vocabulary.endToken) {
		return;
	}
	NSData *bytes = [_constraint.vocabulary bytesForToken:(NSUInteger)token];
	NFKConstraintState next = _state;
	if ([_constraint advanceState:&next bytes:bytes.bytes length:bytes.length]) {
		_state = next;
	}
}

@end

#pragma mark - Base constraint

@implementation NFKTokenConstraint

- (instancetype)initWithVocabulary:(NFKTokenVocabulary *)vocabulary
{
	self = [super init];
	if (self != nil) {
		_vocabulary = vocabulary;
	}
	return self;
}

- (NFKConstraintState)initialState
{
	NFKConstraintState state;
	memset(&state, 0, sizeof(state));
	return state;
}

- (BOOL)advanceState:(NFKConstraintState *)state byte:(uint8_t)byte
{
	[NSException raise:NSInternalInconsistencyException format:@"a subclass supplies the grammar"];
	return NO;
}

- (BOOL)isCompleteState:(const NFKConstraintState *)state
{
	[NSException raise:NSInternalInconsistencyException format:@"a subclass supplies the grammar"];
	return NO;
}

- (BOOL)advanceState:(NFKConstraintState *)state bytes:(const uint8_t *)bytes length:(NSUInteger)length
{
	for (NSUInteger i = 0; i < length; i++) {
		if (![self advanceState:state byte:bytes[i]]) {
			return NO;
		}
	}
	return YES;
}

- (BOOL)stateAfterText:(NSString *)text into:(NFKConstraintState *)outState
{
	NSData *data = [text dataUsingEncoding:NSUTF8StringEncoding];
	NFKConstraintState state = [self initialState];
	if (![self advanceState:&state bytes:data.bytes length:data.length]) {
		return NO;
	}
	*outState = state;
	return YES;
}

- (BOOL)acceptsText:(NSString *)text
{
	NFKConstraintState state;
	return [self stateAfterText:text into:&state];
}

- (BOOL)isCompleteText:(NSString *)text
{
	NFKConstraintState state;
	return [self stateAfterText:text into:&state] && [self isCompleteState:&state];
}

- (NSArray<NSNumber *> *)allowedTokensAfterText:(NSString *)text
{
	NFKConstraintState state;
	if (![self stateAfterText:text into:&state]) {
		return @[];
	}
	NSMutableArray<NSNumber *> *allowed = [NSMutableArray array];
	for (NSUInteger token = 0; token < _vocabulary.count; token++) {
		if ((NSInteger)token == _vocabulary.endToken) {
			if ([self isCompleteState:&state]) {
				[allowed addObject:@(token)];
			}
			continue;
		}
		NSData *bytes = [_vocabulary bytesForToken:token];
		NFKConstraintState trial = state;
		if (bytes.length > 0 && [self advanceState:&trial bytes:bytes.bytes length:bytes.length]) {
			[allowed addObject:@(token)];
		}
	}
	return allowed;
}

- (NFKTokenConstraintCursor *)makeCursor
{
	return [[NFKTokenConstraintCursor alloc] initWithConstraint:self];
}

@end

#pragma mark - JSON

// Frame kinds and phases, packed one byte per nesting level.
enum { NFKJSONFrameObject = 0x10, NFKJSONFrameArray = 0x20 };
enum { NFKJSONObjectKeyOrEnd = 0, NFKJSONObjectKey, NFKJSONObjectColon, NFKJSONObjectValue, NFKJSONObjectCommaOrEnd };
enum { NFKJSONArrayValueOrEnd = 0, NFKJSONArrayValue, NFKJSONArrayCommaOrEnd };
enum { NFKJSONScalarNone = 0, NFKJSONScalarString, NFKJSONScalarNumber, NFKJSONScalarLiteral };
enum { NFKJSONStringBody = 0, NFKJSONStringEscape, NFKJSONStringUnicode0, NFKJSONStringUnicode1, NFKJSONStringUnicode2, NFKJSONStringUnicode3 };
enum { NFKJSONNumberMinus = 0, NFKJSONNumberZero, NFKJSONNumberInteger, NFKJSONNumberDot, NFKJSONNumberFraction,
	   NFKJSONNumberExponent, NFKJSONNumberExponentSign, NFKJSONNumberExponentDigits };
enum { NFKJSONLiteralTrue = 0, NFKJSONLiteralFalse, NFKJSONLiteralNull };

static const char *const NFKJSONLiteralWords[] = { "true", "false", "null" };

static inline BOOL NFKJSONIsWhitespace(uint8_t byte) { return byte == ' ' || byte == '\n' || byte == '\r' || byte == '\t'; }
static inline BOOL NFKJSONIsDigit(uint8_t byte) { return byte >= '0' && byte <= '9'; }
static inline BOOL NFKJSONIsExponent(uint8_t byte) { return byte == 'e' || byte == 'E'; }
static inline BOOL NFKJSONIsHexDigit(uint8_t byte) {
	return NFKJSONIsDigit(byte) || (byte >= 'a' && byte <= 'f') || (byte >= 'A' && byte <= 'F');
}
static inline BOOL NFKJSONNumberIsTerminal(uint8_t phase) {
	return phase == NFKJSONNumberZero || phase == NFKJSONNumberInteger || phase == NFKJSONNumberFraction
		|| phase == NFKJSONNumberExponentDigits;
}

@implementation NFKJSONConstraint

- (instancetype)initWithVocabulary:(NFKTokenVocabulary *)vocabulary root:(NFKJSONRoot)root
{
	return [self initWithVocabulary:vocabulary root:root maximumWhitespaceRun:8];
}

- (instancetype)initWithVocabulary:(NFKTokenVocabulary *)vocabulary
							  root:(NFKJSONRoot)root
			  maximumWhitespaceRun:(NSUInteger)maximumWhitespaceRun
{
	self = [super initWithVocabulary:vocabulary];
	if (self != nil) {
		_root = root;
		_maximumWhitespaceRun = maximumWhitespaceRun;
	}
	return self;
}

- (BOOL)isCompleteState:(const NFKConstraintState *)state
{
	if (state->complete) {
		return YES;
	}
	// A root number cannot know it has ended until something follows; at the end, a terminal digit
	// run is a complete value.
	if (_root != NFKJSONRootAny || state->depth != 0 || state->inKey) {
		return NO;
	}
	return state->scalar == NFKJSONScalarNumber && NFKJSONNumberIsTerminal(state->phase);
}

- (BOOL)advanceState:(NFKConstraintState *)s byte:(uint8_t)byte
{
	// Whitespace between tokens is counted and capped; anything else resets the count. Inside a
	// string the bytes are content, not spacing, and are not counted.
	if (s->scalar != NFKJSONScalarString) {
		if (NFKJSONIsWhitespace(byte)) {
			if (s->whitespaceRun >= _maximumWhitespaceRun) {
				return NO;
			}
			s->whitespaceRun += 1;
		} else {
			s->whitespaceRun = 0;
		}
	}
	return [self advanceValue:s byte:byte];
}

// A value (or a key) has closed: a key waits for its colon, a root value completes the document.
static void NFKJSONFinished(NFKConstraintState *s)
{
	if (s->inKey) {
		s->inKey = 0;
		return;
	}
	if (s->depth == 0) {
		s->complete = 1;
	}
}

// Opens the value `byte` begins, with the enclosing frame already advanced past it.
static BOOL NFKJSONStartValue(NFKConstraintState *s, uint8_t byte, BOOL allowScalar)
{
	switch (byte) {
	case '{':
		if (s->depth >= NFK_CONSTRAINT_MAX_DEPTH) { return NO; }
		s->frames[s->depth++] = NFKJSONFrameObject | NFKJSONObjectKeyOrEnd;
		return YES;
	case '[':
		if (s->depth >= NFK_CONSTRAINT_MAX_DEPTH) { return NO; }
		s->frames[s->depth++] = NFKJSONFrameArray | NFKJSONArrayValueOrEnd;
		return YES;
	case '"':
		if (!allowScalar) { return NO; }
		s->scalar = NFKJSONScalarString; s->phase = NFKJSONStringBody;
		return YES;
	case '-':
		if (!allowScalar) { return NO; }
		s->scalar = NFKJSONScalarNumber; s->phase = NFKJSONNumberMinus;
		return YES;
	case '0':
		if (!allowScalar) { return NO; }
		s->scalar = NFKJSONScalarNumber; s->phase = NFKJSONNumberZero;
		return YES;
	case 't': case 'f': case 'n':
		if (!allowScalar) { return NO; }
		s->scalar = NFKJSONScalarLiteral;
		s->literal = byte == 't' ? NFKJSONLiteralTrue : byte == 'f' ? NFKJSONLiteralFalse : NFKJSONLiteralNull;
		s->phase = 1;
		return YES;
	default:
		if (allowScalar && byte >= '1' && byte <= '9') {
			s->scalar = NFKJSONScalarNumber; s->phase = NFKJSONNumberInteger;
			return YES;
		}
		return NO;
	}
}

static BOOL NFKJSONAdvanceNumber(uint8_t *phase, uint8_t byte)
{
	BOOL digit = NFKJSONIsDigit(byte);
	switch (*phase) {
	case NFKJSONNumberMinus:
		if (byte == '0') { *phase = NFKJSONNumberZero; return YES; }
		if (digit) { *phase = NFKJSONNumberInteger; return YES; }
		return NO;
	case NFKJSONNumberZero:
		if (byte == '.') { *phase = NFKJSONNumberDot; return YES; }
		if (NFKJSONIsExponent(byte)) { *phase = NFKJSONNumberExponent; return YES; }
		return NO;
	case NFKJSONNumberInteger:
		if (digit) { return YES; }
		if (byte == '.') { *phase = NFKJSONNumberDot; return YES; }
		if (NFKJSONIsExponent(byte)) { *phase = NFKJSONNumberExponent; return YES; }
		return NO;
	case NFKJSONNumberDot:
		if (digit) { *phase = NFKJSONNumberFraction; return YES; }
		return NO;
	case NFKJSONNumberFraction:
		if (digit) { return YES; }
		if (NFKJSONIsExponent(byte)) { *phase = NFKJSONNumberExponent; return YES; }
		return NO;
	case NFKJSONNumberExponent:
		if (byte == '+' || byte == '-') { *phase = NFKJSONNumberExponentSign; return YES; }
		if (digit) { *phase = NFKJSONNumberExponentDigits; return YES; }
		return NO;
	default:	// exponent sign or digits
		if (digit) { *phase = NFKJSONNumberExponentDigits; return YES; }
		return NO;
	}
}

- (BOOL)advanceString:(NFKConstraintState *)s byte:(uint8_t)byte
{
	switch (s->phase) {
	case NFKJSONStringBody:
		if (byte == '"') { s->scalar = NFKJSONScalarNone; NFKJSONFinished(s); return YES; }
		if (byte == '\\') { s->phase = NFKJSONStringEscape; return YES; }
		return byte >= 0x20;
	case NFKJSONStringEscape:
		if (byte == 'u') { s->phase = NFKJSONStringUnicode0; return YES; }
		if (strchr("\"\\/bfnrt", byte) != NULL && byte != 0) { s->phase = NFKJSONStringBody; return YES; }
		return NO;
	default:	// unicode digits
		if (!NFKJSONIsHexDigit(byte)) { return NO; }
		s->phase = s->phase == NFKJSONStringUnicode3 ? NFKJSONStringBody : s->phase + 1;
		return YES;
	}
}

- (BOOL)advanceValue:(NFKConstraintState *)s byte:(uint8_t)byte
{
	if (s->complete) {
		return NFKJSONIsWhitespace(byte);
	}
	switch (s->scalar) {
	case NFKJSONScalarString:
		return [self advanceString:s byte:byte];
	case NFKJSONScalarNumber: {
		uint8_t phase = s->phase;
		if (NFKJSONAdvanceNumber(&phase, byte)) { s->phase = phase; return YES; }
		if (!NFKJSONNumberIsTerminal(s->phase)) { return NO; }
		s->scalar = NFKJSONScalarNone;
		NFKJSONFinished(s);
		return [self advanceValue:s byte:byte];
	}
	case NFKJSONScalarLiteral: {
		const char *word = NFKJSONLiteralWords[s->literal];
		if ((uint8_t)word[s->phase] != byte) { return NO; }
		if (word[s->phase + 1] == '\0') { s->scalar = NFKJSONScalarNone; NFKJSONFinished(s); return YES; }
		s->phase += 1;
		return YES;
	}
	default:
		break;
	}

	if (NFKJSONIsWhitespace(byte)) {
		return YES;
	}
	if (s->depth == 0) {
		if ((_root == NFKJSONRootObject && byte != '{') || (_root == NFKJSONRootArray && byte != '[')) {
			return NO;
		}
		return NFKJSONStartValue(s, byte, _root == NFKJSONRootAny);
	}
	uint8_t *top = &s->frames[s->depth - 1];
	uint8_t kind = *top & 0xF0, phase = *top & 0x0F;
	if (kind == NFKJSONFrameObject) {
		if ((phase == NFKJSONObjectKeyOrEnd || phase == NFKJSONObjectCommaOrEnd) && byte == '}') {
			s->depth -= 1; NFKJSONFinished(s); return YES;
		}
		if ((phase == NFKJSONObjectKeyOrEnd || phase == NFKJSONObjectKey) && byte == '"') {
			*top = NFKJSONFrameObject | NFKJSONObjectColon;
			s->scalar = NFKJSONScalarString; s->phase = NFKJSONStringBody; s->inKey = 1;
			return YES;
		}
		if (phase == NFKJSONObjectColon && byte == ':') { *top = NFKJSONFrameObject | NFKJSONObjectValue; return YES; }
		if (phase == NFKJSONObjectValue) {
			*top = NFKJSONFrameObject | NFKJSONObjectCommaOrEnd;
			return NFKJSONStartValue(s, byte, YES);
		}
		if (phase == NFKJSONObjectCommaOrEnd && byte == ',') { *top = NFKJSONFrameObject | NFKJSONObjectKey; return YES; }
		return NO;
	}
	if ((phase == NFKJSONArrayValueOrEnd || phase == NFKJSONArrayCommaOrEnd) && byte == ']') {
		s->depth -= 1; NFKJSONFinished(s); return YES;
	}
	if (phase == NFKJSONArrayValueOrEnd || phase == NFKJSONArrayValue) {
		*top = NFKJSONFrameArray | NFKJSONArrayCommaOrEnd;
		return NFKJSONStartValue(s, byte, YES);
	}
	if (phase == NFKJSONArrayCommaOrEnd && byte == ',') { *top = NFKJSONFrameArray | NFKJSONArrayValue; return YES; }
	return NO;
}

@end

#pragma mark - Choices

@implementation NFKChoiceConstraint
{
	NSArray<NSData *> *_choiceBytes;
	NSMutableData *_emitted;		// scratch: the bytes a trial has walked, keyed by state.length
}

- (instancetype)initWithChoices:(NSArray<NSString *> *)choices vocabulary:(NFKTokenVocabulary *)vocabulary
{
	self = [super initWithVocabulary:vocabulary];
	if (self != nil) {
		_choices = [choices copy];
		NSMutableArray<NSData *> *bytes = [NSMutableArray arrayWithCapacity:choices.count];
		for (NSString *choice in choices) {
			[bytes addObject:[choice dataUsingEncoding:NSUTF8StringEncoding]];
		}
		_choiceBytes = bytes;
	}
	return self;
}

// The state is the number of bytes emitted; the bytes themselves are what every choice must share
// up to that length, so advancing checks that some choice carries `byte` at position `length`
// among those that agree with the run so far. Since every candidate agrees on the emitted prefix,
// the emitted bytes are recoverable from any candidate — `_emitted` is not needed beyond the length.
- (BOOL)advanceState:(NFKConstraintState *)state byte:(uint8_t)byte
{
	NSUInteger position = state->length;
	for (NSData *choice in _choiceBytes) {
		if (choice.length > position && ((const uint8_t *)choice.bytes)[position] == byte
			&& [self choice:choice agreesWithRunOfLength:position state:state]) {
			state->length = (uint32_t)(position + 1);
			// Remember which choice the run follows, so later bytes are checked against the same prefix.
			state->literal = 1;
			return YES;
		}
	}
	return NO;
}

// Every candidate that survived so far shares the emitted prefix; `frames` records the chosen path's
// bytes (up to the depth limit) so a divergence between two choices with a common prefix is caught.
- (BOOL)choice:(NSData *)choice agreesWithRunOfLength:(NSUInteger)length state:(NFKConstraintState *)state
{
	const uint8_t *bytes = choice.bytes;
	NSUInteger tracked = MIN(length, (NSUInteger)NFK_CONSTRAINT_MAX_DEPTH);
	for (NSUInteger i = 0; i < tracked; i++) {
		if (state->frames[i] != bytes[i]) {
			return NO;
		}
	}
	if (length < NFK_CONSTRAINT_MAX_DEPTH) {
		state->frames[length] = bytes[length];
	}
	return YES;
}

- (BOOL)isCompleteState:(const NFKConstraintState *)state
{
	for (NSData *choice in _choiceBytes) {
		if (choice.length == state->length) {
			const uint8_t *bytes = choice.bytes;
			BOOL same = YES;
			NSUInteger tracked = MIN((NSUInteger)state->length, (NSUInteger)NFK_CONSTRAINT_MAX_DEPTH);
			for (NSUInteger i = 0; i < tracked; i++) {
				if (state->frames[i] != bytes[i]) { same = NO; break; }
			}
			if (same) {
				return YES;
			}
		}
	}
	return NO;
}

@end
