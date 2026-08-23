//
//  NFKUnigramTokenizer.m
//  InferKit
//

#import "NFKUnigramTokenizer.h"
#import "NFKTokenizerPrivate.h"
#import "NFKErrors.h"

// SentencePiece renders a space as U+2581 (LOWER ONE EIGHTH BLOCK).
static NSString * const kNFKUnigramSpace = @"▁";

// A fallback step is only taken when no vocabulary piece covers a character, so it must lose to any
// real segmentation while still keeping every position reachable.
static const double kNFKUnigramFallbackScore = -1.0e4;

@implementation NFKUnigramTokenizer
{
	NSArray<NSString *> *_pieces;					// id -> piece
	NSArray<NSNumber *> *_scores;					// id -> score
	NSDictionary<NSString *, NSNumber *> *_pieceToId;
	NSDictionary<NSNumber *, NSNumber *> *_byteTokenIds;	// byte value -> id for <0xHH> pieces
	NSDictionary<NSString *, NSNumber *> *_specialTokens;
	NSInteger _unkId;
	BOOL _byteFallback;
	BOOL _addDummyPrefix;
	NSUInteger _maxPieceLength;
}

- (nullable instancetype)initWithVocabURL:(NSURL *)vocabURL
							specialTokens:(nullable NSDictionary<NSString *, NSNumber *> *)specialTokens
									  eos:(NSInteger)eosTokenId
									  bos:(NSInteger)bosTokenId
									error:(NSError * _Nullable *)outError
{
	self = [super initWithEOS:eosTokenId bos:bosTokenId];
	if (self == nil) {
		return nil;
	}

	NSData *data = [NSData dataWithContentsOfURL:vocabURL options:0 error:outError];
	if (data == nil) {
		return nil;
	}
	id json = [NSJSONSerialization JSONObjectWithData:data options:0 error:outError];
	if (![json isKindOfClass:NSDictionary.class] || ![json[@"vocab"] isKindOfClass:NSArray.class]) {
		[NFKTokenizer setError:outError code:kNFKError_InferenceBackendFailure reason:@"the unigram vocab file is malformed"];
		return nil;
	}

	if (![self loadVocab:json[@"vocab"] error:outError]) {
		return nil;
	}
	_unkId = [json[@"unkId"] isKindOfClass:NSNumber.class] ? [json[@"unkId"] integerValue] : 0;
	_byteFallback = [json[@"byteFallback"] boolValue];
	_addDummyPrefix = json[@"addDummyPrefix"] != nil ? [json[@"addDummyPrefix"] boolValue] : YES;
	_specialTokens = [specialTokens copy];

	return self;
}

- (BOOL)loadVocab:(NSArray *)vocab error:(NSError * _Nullable *)outError
{
	NSMutableArray<NSString *> *pieces = [NSMutableArray arrayWithCapacity:vocab.count];
	NSMutableArray<NSNumber *> *scores = [NSMutableArray arrayWithCapacity:vocab.count];
	NSMutableDictionary<NSString *, NSNumber *> *pieceToId = [NSMutableDictionary dictionaryWithCapacity:vocab.count];
	NSMutableDictionary<NSNumber *, NSNumber *> *byteTokenIds = [NSMutableDictionary dictionary];
	NSUInteger maxLength = 1;

	for (NSUInteger identifier = 0; identifier < vocab.count; identifier++) {
		id entry = vocab[identifier];
		if (![entry isKindOfClass:NSArray.class] || [entry count] < 1 || ![entry[0] isKindOfClass:NSString.class]) {
			[NFKTokenizer setError:outError code:kNFKError_InferenceBackendFailure reason:@"a unigram vocab entry is malformed"];
			return NO;
		}
		NSString *piece = entry[0];
		double score = [entry count] >= 2 ? [entry[1] doubleValue] : 0.0;
		pieces[identifier] = piece;
		scores[identifier] = @(score);
		if (pieceToId[piece] == nil) {
			pieceToId[piece] = @(identifier);
		}
		maxLength = MAX(maxLength, piece.length);

		NSInteger byteValue = [self byteValueForPiece:piece];
		if (byteValue >= 0) {
			byteTokenIds[@(byteValue)] = @(identifier);
		}
	}

	_pieces = [pieces copy];
	_scores = [scores copy];
	_pieceToId = [pieceToId copy];
	_byteTokenIds = [byteTokenIds copy];
	_maxPieceLength = maxLength;
	return YES;
}

// The byte value of a "<0xHH>" fallback piece, or -1 when the piece is not one.
- (NSInteger)byteValueForPiece:(NSString *)piece
{
	if (piece.length != 6 || ![piece hasPrefix:@"<0x"] || ![piece hasSuffix:@">"]) {
		return -1;
	}
	unsigned int value = 0;
	NSScanner *scanner = [NSScanner scannerWithString:[piece substringWithRange:NSMakeRange(3, 2)]];
	if (![scanner scanHexInt:&value] || !scanner.isAtEnd) {
		return -1;
	}
	return (NSInteger)value;
}

#pragma mark Encoding

- (NSArray<NSNumber *> *)encode:(NSString *)text
{
	NSMutableArray<NSNumber *> *ids = [NSMutableArray array];
	for (NSString *segment in [self segmentsOfText:text bySpecialTokens:_specialTokens]) {
		NSNumber *special = _specialTokens[segment];
		if (special != nil) {
			[ids addObject:special];
			continue;
		}
		[self encodeSegment:[self normalize:segment] into:ids];
	}
	return ids;
}

- (NSString *)normalize:(NSString *)text
{
	if (text.length == 0) {
		return text;
	}
	NSString *prefixed = _addDummyPrefix ? [@" " stringByAppendingString:text] : text;
	return [prefixed stringByReplacingOccurrencesOfString:@" " withString:kNFKUnigramSpace];
}

- (void)encodeSegment:(NSString *)text into:(NSMutableArray<NSNumber *> *)ids
{
	NSUInteger count = text.length;
	if (count == 0) {
		return;
	}

	double *best = malloc(sizeof(double) * (count + 1));
	NSInteger *backLength = malloc(sizeof(NSInteger) * (count + 1));
	NSInteger *backId = malloc(sizeof(NSInteger) * (count + 1));		// -1 marks a byte-fallback step
	for (NSUInteger i = 0; i <= count; i++) {
		best[i] = -INFINITY;
		backLength[i] = 0;
	}
	best[0] = 0.0;

	for (NSUInteger i = 0; i < count; i++) {
		if (best[i] == -INFINITY) {
			continue;
		}
		NSUInteger charLength = [text rangeOfComposedCharacterSequenceAtIndex:i].length;

		NSUInteger maxLength = MIN(_maxPieceLength, count - i);
		for (NSUInteger length = charLength; length <= maxLength; length++) {
			NSNumber *identifier = _pieceToId[[text substringWithRange:NSMakeRange(i, length)]];
			if (identifier == nil) {
				continue;
			}
			double candidate = best[i] + _scores[identifier.integerValue].doubleValue;
			NSUInteger end = i + length;
			if (candidate > best[end]) {
				best[end] = candidate;
				backLength[end] = (NSInteger)length;
				backId[end] = identifier.integerValue;
			}
		}

		NSUInteger fallbackEnd = i + charLength;
		double fallback = best[i] + kNFKUnigramFallbackScore;
		if (fallback > best[fallbackEnd]) {
			best[fallbackEnd] = fallback;
			backLength[fallbackEnd] = (NSInteger)charLength;
			backId[fallbackEnd] = -1;
		}
	}

	[self backtrackFrom:count text:text length:backLength identifier:backId into:ids];

	free(best);
	free(backLength);
	free(backId);
}

- (void)backtrackFrom:(NSUInteger)count
				 text:(NSString *)text
			   length:(const NSInteger *)backLength
		   identifier:(const NSInteger *)backId
				 into:(NSMutableArray<NSNumber *> *)ids
{
	NSMutableArray<NSArray<NSNumber *> *> *stepsReversed = [NSMutableArray array];
	NSUInteger position = count;
	while (position > 0) {
		NSInteger length = backLength[position];
		if (length <= 0) {
			break;
		}
		[stepsReversed addObject:@[@(position - length), @(length), @(backId[position])]];
		position -= length;
	}

	for (NSArray<NSNumber *> *step in [stepsReversed reverseObjectEnumerator]) {
		NSInteger identifier = step[2].integerValue;
		if (identifier >= 0) {
			[ids addObject:@(identifier)];
		} else {
			NSRange range = NSMakeRange(step[0].unsignedIntegerValue, step[1].unsignedIntegerValue);
			[self appendFallbackFor:[text substringWithRange:range] into:ids];
		}
	}
}

- (void)appendFallbackFor:(NSString *)text into:(NSMutableArray<NSNumber *> *)ids
{
	if (!_byteFallback) {
		[ids addObject:@(_unkId)];
		return;
	}
	NSData *bytes = [text dataUsingEncoding:NSUTF8StringEncoding];
	const unsigned char *values = bytes.bytes;
	for (NSUInteger i = 0; i < bytes.length; i++) {
		NSNumber *identifier = _byteTokenIds[@(values[i])];
		[ids addObject:identifier != nil ? identifier : @(_unkId)];
	}
}

#pragma mark Decoding

- (NSString *)decode:(NSArray<NSNumber *> *)tokenIds
{
	NSMutableString *out = [NSMutableString string];
	NSMutableData *byteRun = [NSMutableData data];

	for (NSNumber *tokenId in tokenIds) {
		NSString *piece = [self pieceForId:tokenId];
		if (piece == nil) {
			continue;
		}
		NSInteger byteValue = [self byteValueForPiece:piece];
		if (byteValue >= 0) {
			unsigned char byte = (unsigned char)byteValue;
			[byteRun appendBytes:&byte length:1];
		} else {
			[self flushBytes:byteRun into:out];
			[out appendString:piece];
		}
	}
	[self flushBytes:byteRun into:out];

	NSString *text = [out stringByReplacingOccurrencesOfString:kNFKUnigramSpace withString:@" "];
	if (_addDummyPrefix && [text hasPrefix:@" "]) {
		text = [text substringFromIndex:1];
	}
	return text;
}

- (nullable NSString *)pieceForId:(NSNumber *)tokenId
{
	NSInteger identifier = tokenId.integerValue;
	if (identifier >= 0 && (NSUInteger)identifier < _pieces.count) {
		return _pieces[identifier];
	}
	for (NSString *literal in _specialTokens) {
		if ([_specialTokens[literal] isEqualToNumber:tokenId]) {
			return literal;
		}
	}
	return nil;
}

- (void)flushBytes:(NSMutableData *)byteRun into:(NSMutableString *)out
{
	if (byteRun.length == 0) {
		return;
	}
	NSString *text = [[NSString alloc] initWithData:byteRun encoding:NSUTF8StringEncoding];
	if (text != nil) {
		[out appendString:text];
	}
	byteRun.length = 0;
}

@end
