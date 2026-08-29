//
//  NFKByteLevelBPETokenizer.m
//  InferKit
//

#import "NFKByteLevelBPETokenizer.h"
#import "NFKTokenizerPrivate.h"
#import "NFKErrors.h"

// GPT-2 pre-tokenization: contractions, then runs of letters, digits, other non-space, and
// whitespace, each optionally led by a single space.
static NSString * const kNFKBPEPattern =
	@"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

// Qwen2 pre-tokenization, from the released tokenizer.json: case-insensitive contractions, a letter
// run optionally led by ONE non-letter/digit character, single digits, a punctuation run absorbing
// trailing newlines, and whitespace runs ending in a newline held together.
static NSString * const kNFKQwen2BPEPattern =
	@"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

@implementation NFKByteLevelBPETokenizer
{
	NSDictionary<NSString *, NSNumber *> *_encoder;			// token string -> id
	NSDictionary<NSNumber *, NSString *> *_decoder;			// id -> token string
	NSDictionary<NSString *, NSNumber *> *_ranks;			// "first\nsecond" -> merge rank
	NSDictionary<NSString *, NSNumber *> *_specialTokens;	// literal -> id
	NSString *_pretokenization;								// "gpt2" (nil) or "qwen2"
	NSRegularExpression *_pattern;
	unichar _byteToUnicode[256];
	NSDictionary<NSNumber *, NSNumber *> *_unicodeToByte;	// unichar value -> byte
}

- (nullable instancetype)initWithVocabURL:(NSURL *)vocabURL
								mergesURL:(NSURL *)mergesURL
							specialTokens:(nullable NSDictionary<NSString *, NSNumber *> *)specialTokens
									  eos:(NSInteger)eosTokenId
									  bos:(NSInteger)bosTokenId
									error:(NSError * _Nullable *)outError
{
	return [self initWithVocabURL:vocabURL
						mergesURL:mergesURL
					specialTokens:specialTokens
				  pretokenization:nil
							  eos:eosTokenId
							  bos:bosTokenId
							error:outError];
}

- (nullable instancetype)initWithVocabURL:(NSURL *)vocabURL
								mergesURL:(NSURL *)mergesURL
							specialTokens:(nullable NSDictionary<NSString *, NSNumber *> *)specialTokens
						  pretokenization:(nullable NSString *)pretokenization
									  eos:(NSInteger)eosTokenId
									  bos:(NSInteger)bosTokenId
									error:(NSError * _Nullable *)outError
{
	self = [super initWithEOS:eosTokenId bos:bosTokenId];
	if (self == nil) {
		return nil;
	}
	if (pretokenization != nil
		&& ![pretokenization isEqualToString:@"gpt2"] && ![pretokenization isEqualToString:@"qwen2"]) {
		[NFKTokenizer setError:outError code:kNFKError_InferenceUnsupported
						reason:[NSString stringWithFormat:@"unknown pretokenization \"%@\"", pretokenization]];
		return nil;
	}
	_pretokenization = [pretokenization copy];

	if (![self loadEncoderFromURL:vocabURL error:outError]) {
		return nil;
	}
	if (![self loadRanksFromURL:mergesURL error:outError]) {
		return nil;
	}

	_specialTokens = [specialTokens copy];
	[self buildByteMaps];

	NSError *patternError = nil;
	_pattern = [NSRegularExpression regularExpressionWithPattern:[self pretokenizationPattern] options:0 error:&patternError];
	if (_pattern == nil) {
		if (outError != NULL) {
			*outError = patternError;
		}
		return nil;
	}

	return self;
}

#pragma mark Loading

- (BOOL)loadEncoderFromURL:(NSURL *)url error:(NSError * _Nullable *)outError
{
	NSData *data = [NSData dataWithContentsOfURL:url options:0 error:outError];
	if (data == nil) {
		return NO;
	}
	id json = [NSJSONSerialization JSONObjectWithData:data options:0 error:outError];
	if (![json isKindOfClass:NSDictionary.class]) {
		[NFKTokenizer setError:outError code:kNFKError_InferenceBackendFailure reason:@"vocab.json is not a JSON object"];
		return NO;
	}
	_encoder = [json copy];

	NSMutableDictionary<NSNumber *, NSString *> *decoder = [NSMutableDictionary dictionaryWithCapacity:_encoder.count];
	[_encoder enumerateKeysAndObjectsUsingBlock:^(NSString *token, NSNumber *tokenId, BOOL *stop) {
		decoder[tokenId] = token;
	}];
	_decoder = [decoder copy];
	return YES;
}

- (BOOL)loadRanksFromURL:(NSURL *)url error:(NSError * _Nullable *)outError
{
	NSString *text = [NSString stringWithContentsOfURL:url encoding:NSUTF8StringEncoding error:outError];
	if (text == nil) {
		return NO;
	}
	NSMutableDictionary<NSString *, NSNumber *> *ranks = [NSMutableDictionary dictionary];
	__block NSInteger rank = 0;
	[text enumerateLinesUsingBlock:^(NSString *line, BOOL *stop) {
		if (line.length == 0 || [line hasPrefix:@"#"]) {
			return;
		}
		NSArray<NSString *> *parts = [line componentsSeparatedByString:@" "];
		if (parts.count != 2) {
			return;
		}
		ranks[[self pairKeyForFirst:parts[0] second:parts[1]]] = @(rank);
		rank += 1;
	}];
	_ranks = [ranks copy];
	return YES;
}

// A byte's mapped character is never U+000A, so newline separates the pair halves unambiguously.
- (NSString *)pairKeyForFirst:(NSString *)first second:(NSString *)second
{
	return [NSString stringWithFormat:@"%@\n%@", first, second];
}

// GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to code points above 255,
// so every byte becomes a character the merges can operate on.
- (void)buildByteMaps
{
	NSMutableDictionary<NSNumber *, NSNumber *> *unicodeToByte = [NSMutableDictionary dictionaryWithCapacity:256];
	int shifted = 0;
	for (int b = 0; b < 256; b++) {
		BOOL printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
		unichar mapped;
		if (printable) {
			mapped = (unichar)b;
		} else {
			mapped = (unichar)(256 + shifted);
			shifted += 1;
		}
		_byteToUnicode[b] = mapped;
		unicodeToByte[@(mapped)] = @(b);
	}
	_unicodeToByte = [unicodeToByte copy];
}

#pragma mark Subclass hooks

- (NSString *)pretokenizationPattern
{
	if ([_pretokenization isEqualToString:@"qwen2"]) {
		return kNFKQwen2BPEPattern;
	}
	return kNFKBPEPattern;
}

- (NSString *)normalizedText:(NSString *)text
{
	return text;
}

- (NSArray<NSString *> *)symbolsForWord:(NSString *)word
{
	NSMutableArray<NSString *> *symbols = [NSMutableArray arrayWithCapacity:word.length];
	for (NSUInteger i = 0; i < word.length; i++) {
		[symbols addObject:[word substringWithRange:NSMakeRange(i, 1)]];
	}
	return symbols;
}

- (NSString *)finalizedText:(NSString *)text
{
	return text;
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
		[self encodeSegment:[self normalizedText:segment] into:ids];
	}
	return ids;
}

- (void)encodeSegment:(NSString *)text into:(NSMutableArray<NSNumber *> *)ids
{
	NSArray<NSTextCheckingResult *> *matches =
		[_pattern matchesInString:text options:0 range:NSMakeRange(0, text.length)];
	for (NSTextCheckingResult *match in matches) {
		NSString *piece = [self byteStringForPiece:[text substringWithRange:match.range]];
		for (NSString *token in [self mergedTokensForWord:piece]) {
			NSNumber *tokenId = _encoder[token];
			if (tokenId != nil) {
				[ids addObject:tokenId];
			}
		}
	}
}

- (NSString *)byteStringForPiece:(NSString *)piece
{
	NSData *data = [piece dataUsingEncoding:NSUTF8StringEncoding];
	const unsigned char *bytes = data.bytes;
	unichar *characters = malloc(sizeof(unichar) * data.length);
	for (NSUInteger i = 0; i < data.length; i++) {
		characters[i] = _byteToUnicode[bytes[i]];
	}
	NSString *result = [NSString stringWithCharacters:characters length:data.length];
	free(characters);
	return result;
}

- (NSArray<NSString *> *)mergedTokensForWord:(NSString *)word
{
	NSMutableArray<NSString *> *symbols = [[self symbolsForWord:word] mutableCopy];

	while (symbols.count >= 2) {
		NSInteger bestRank = NSIntegerMax;
		NSUInteger bestIndex = NSNotFound;
		for (NSUInteger i = 0; i + 1 < symbols.count; i++) {
			NSNumber *rank = _ranks[[self pairKeyForFirst:symbols[i] second:symbols[i + 1]]];
			if (rank != nil && rank.integerValue < bestRank) {
				bestRank = rank.integerValue;
				bestIndex = i;
			}
		}
		if (bestIndex == NSNotFound) {
			break;
		}

		NSString *first = symbols[bestIndex];
		NSString *second = symbols[bestIndex + 1];
		NSMutableArray<NSString *> *merged = [NSMutableArray arrayWithCapacity:symbols.count];
		NSUInteger i = 0;
		while (i < symbols.count) {
			if (i + 1 < symbols.count && [symbols[i] isEqualToString:first] && [symbols[i + 1] isEqualToString:second]) {
				[merged addObject:[first stringByAppendingString:second]];
				i += 2;
			} else {
				[merged addObject:symbols[i]];
				i += 1;
			}
		}
		symbols = merged;
	}
	return symbols;
}

#pragma mark Decoding

- (NSString *)decode:(NSArray<NSNumber *> *)tokenIds
{
	NSMutableString *byteString = [NSMutableString string];
	for (NSNumber *tokenId in tokenIds) {
		NSString *token = _decoder[tokenId];
		if (token == nil) {
			token = [self specialTokenForId:tokenId];
		}
		if (token != nil) {
			[byteString appendString:token];
		}
	}

	NSUInteger length = byteString.length;
	unsigned char *bytes = malloc(length > 0 ? length : 1);
	NSUInteger count = 0;
	for (NSUInteger i = 0; i < length; i++) {
		NSNumber *byte = _unicodeToByte[@([byteString characterAtIndex:i])];
		if (byte != nil) {
			bytes[count++] = (unsigned char)byte.integerValue;
		}
	}
	NSString *text = [[NSString alloc] initWithBytes:bytes length:count encoding:NSUTF8StringEncoding];
	free(bytes);
	return text != nil ? [self finalizedText:text] : @"";
}

- (nullable NSString *)specialTokenForId:(NSNumber *)tokenId
{
	for (NSString *literal in _specialTokens) {
		if ([_specialTokens[literal] isEqualToNumber:tokenId]) {
			return literal;
		}
	}
	return nil;
}

@end
