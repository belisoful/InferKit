//
//  NFKWordPieceTokenizer.m
//  InferKit
//

#import "NFKWordPieceTokenizer.h"
#import "NFKTokenizerPrivate.h"
#import "NFKErrors.h"

@implementation NFKWordPieceTokenizer
{
	NSDictionary<NSString *, NSNumber *> *_vocab;			// token -> id
	NSArray<NSString *> *_tokensById;						// id -> token
	NSDictionary<NSString *, NSNumber *> *_specialTokens;
	NSString *_unkToken;
	NSInteger _unkId;
	NSString *_continuingPrefix;
	BOOL _lowercase;
	NSUInteger _maxCharsPerWord;
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
	if (![json isKindOfClass:NSDictionary.class] || ![json[@"vocab"] isKindOfClass:NSDictionary.class]) {
		[NFKTokenizer setError:outError code:kNFKError_InferenceBackendFailure reason:@"the wordpiece vocab file is malformed"];
		return nil;
	}

	_vocab = [json[@"vocab"] copy];
	_tokensById = [self tokensByIdFromVocab:_vocab];
	_unkToken = [json[@"unkToken"] isKindOfClass:NSString.class] ? json[@"unkToken"] : @"[UNK]";
	_unkId = [_vocab[_unkToken] integerValue];
	_continuingPrefix = [json[@"continuingSubwordPrefix"] isKindOfClass:NSString.class] ? json[@"continuingSubwordPrefix"] : @"##";
	_lowercase = json[@"lowercase"] != nil ? [json[@"lowercase"] boolValue] : YES;
	_maxCharsPerWord = [json[@"maxCharsPerWord"] isKindOfClass:NSNumber.class] ? [json[@"maxCharsPerWord"] unsignedIntegerValue] : 100;
	_specialTokens = [specialTokens copy];

	return self;
}

- (NSArray<NSString *> *)tokensByIdFromVocab:(NSDictionary<NSString *, NSNumber *> *)vocab
{
	NSMutableArray<NSString *> *tokens = [NSMutableArray arrayWithCapacity:vocab.count];
	for (NSUInteger i = 0; i < vocab.count; i++) {
		[tokens addObject:@""];
	}
	[vocab enumerateKeysAndObjectsUsingBlock:^(NSString *token, NSNumber *identifier, BOOL *stop) {
		NSInteger value = identifier.integerValue;
		if (value >= 0 && (NSUInteger)value < tokens.count) {
			tokens[value] = token;
		}
	}];
	return [tokens copy];
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
		for (NSString *word in [self basicTokenize:segment]) {
			[self wordPieceEncode:word into:ids];
		}
	}
	return ids;
}

// Whitespace and punctuation splitting, with optional lowercasing; punctuation becomes its own word.
- (NSArray<NSString *> *)basicTokenize:(NSString *)text
{
	NSString *prepared = _lowercase ? text.lowercaseString : text;
	NSMutableArray<NSString *> *words = [NSMutableArray array];
	NSMutableString *current = [NSMutableString string];
	NSCharacterSet *whitespace = [NSCharacterSet whitespaceAndNewlineCharacterSet];

	[prepared enumerateSubstringsInRange:NSMakeRange(0, prepared.length)
								options:NSStringEnumerationByComposedCharacterSequences
							 usingBlock:^(NSString *character, NSRange range, NSRange enclosing, BOOL *stop) {
		unichar first = [character characterAtIndex:0];
		if ([whitespace characterIsMember:first]) {
			if (current.length > 0) {
				[words addObject:[current copy]];
				[current setString:@""];
			}
		} else if ([self isPunctuation:character]) {
			if (current.length > 0) {
				[words addObject:[current copy]];
				[current setString:@""];
			}
			[words addObject:character];
		} else {
			[current appendString:character];
		}
	}];
	if (current.length > 0) {
		[words addObject:[current copy]];
	}
	return words;
}

- (BOOL)isPunctuation:(NSString *)character
{
	unichar first = [character characterAtIndex:0];
	BOOL asciiPunctuation = (first >= 33 && first <= 47) || (first >= 58 && first <= 64) ||
							(first >= 91 && first <= 96) || (first >= 123 && first <= 126);
	if (asciiPunctuation) {
		return YES;
	}
	return [[NSCharacterSet punctuationCharacterSet] characterIsMember:first] ||
		   [[NSCharacterSet symbolCharacterSet] characterIsMember:first];
}

- (void)wordPieceEncode:(NSString *)word into:(NSMutableArray<NSNumber *> *)ids
{
	if (word.length == 0) {
		return;
	}
	if (word.length > _maxCharsPerWord) {
		[ids addObject:@(_unkId)];
		return;
	}

	NSMutableArray<NSNumber *> *pieces = [NSMutableArray array];
	NSUInteger start = 0;
	while (start < word.length) {
		NSUInteger end = word.length;
		NSNumber *matchedId = nil;
		while (start < end) {
			NSString *piece = [word substringWithRange:NSMakeRange(start, end - start)];
			if (start > 0) {
				piece = [_continuingPrefix stringByAppendingString:piece];
			}
			NSNumber *identifier = _vocab[piece];
			if (identifier != nil) {
				matchedId = identifier;
				break;
			}
			end -= [word rangeOfComposedCharacterSequenceAtIndex:end - 1].length;
		}
		if (matchedId == nil) {
			[ids addObject:@(_unkId)];		// an unsplittable word maps to a single unknown token
			return;
		}
		[pieces addObject:matchedId];
		start = end;
	}
	[ids addObjectsFromArray:pieces];
}

#pragma mark Decoding

- (NSString *)decode:(NSArray<NSNumber *> *)tokenIds
{
	NSMutableString *out = [NSMutableString string];
	for (NSNumber *tokenId in tokenIds) {
		NSString *token = [self tokenForId:tokenId];
		if (token == nil) {
			continue;
		}
		if ([token hasPrefix:_continuingPrefix]) {
			[out appendString:[token substringFromIndex:_continuingPrefix.length]];
		} else {
			if (out.length > 0) {
				[out appendString:@" "];
			}
			[out appendString:token];
		}
	}
	return out;
}

- (nullable NSData *)bytesForTokenId:(NSInteger)tokenId
{
	NSString *token = [self tokenForId:@(tokenId)];
	if (token == nil) {
		return nil;
	}
	if ([token hasPrefix:_continuingPrefix]) {
		return [[token substringFromIndex:_continuingPrefix.length] dataUsingEncoding:NSUTF8StringEncoding];
	}
	return [[@" " stringByAppendingString:token] dataUsingEncoding:NSUTF8StringEncoding];
}

- (nullable NSString *)tokenForId:(NSNumber *)tokenId
{
	NSInteger identifier = tokenId.integerValue;
	if (identifier >= 0 && (NSUInteger)identifier < _tokensById.count) {
		NSString *token = _tokensById[identifier];
		if (token.length > 0) {
			return token;
		}
	}
	for (NSString *literal in _specialTokens) {
		if ([_specialTokens[literal] isEqualToNumber:tokenId]) {
			return literal;
		}
	}
	return nil;
}

@end
