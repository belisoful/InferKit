//
//  NFKTokenizer.m
//  InferKit
//

#import "NFKTokenizer.h"
#import "NFKTokenizerPrivate.h"
#import "NFKByteLevelBPETokenizer.h"
#import "NFKCLIPTokenizer.h"
#import "NFKUnigramTokenizer.h"
#import "NFKWordPieceTokenizer.h"
#import "NFKErrors.h"

@implementation NFKTokenizer

@synthesize eosTokenId = _eosTokenId;
@synthesize bosTokenId = _bosTokenId;

+ (nullable instancetype)tokenizerForManifest:(NSDictionary<NSString *, id> *)manifest
									directory:(NSURL *)directory
										error:(NSError * _Nullable *)outError
{
	NSDictionary *spec = manifest[@"tokenizer"];
	if (![spec isKindOfClass:NSDictionary.class]) {
		[self setError:outError code:kNFKError_InferenceUnsupported reason:@"the manifest has no tokenizer section"];
		return nil;
	}

	NSInteger eos = [self tokenId:manifest[@"eosTokenId"]];
	NSInteger bos = [self tokenId:manifest[@"bosTokenId"]];
	NSDictionary<NSString *, NSNumber *> *specialTokens =
		[spec[@"specialTokens"] isKindOfClass:NSDictionary.class] ? spec[@"specialTokens"] : nil;

	NSString *type = spec[@"type"];
	if ([type isEqualToString:@"clip"] || [type isEqualToString:@"bpe-clip"]) {
		NSString *vocabName = [spec[@"vocab"] isKindOfClass:NSString.class] ? spec[@"vocab"] : @"vocab.json";
		NSString *mergesName = [spec[@"merges"] isKindOfClass:NSString.class] ? spec[@"merges"] : @"merges.txt";
		return [[NFKCLIPTokenizer alloc] initWithVocabURL:[directory URLByAppendingPathComponent:vocabName]
											   mergesURL:[directory URLByAppendingPathComponent:mergesName]
										   specialTokens:specialTokens
													 eos:eos
													 bos:bos
												   error:outError];
	}
	if ([type isEqualToString:@"bpe-bytelevel"] || [type isEqualToString:@"bytelevel-bpe"] || [type isEqualToString:@"bpe"]) {
		NSString *vocabName = [spec[@"vocab"] isKindOfClass:NSString.class] ? spec[@"vocab"] : @"vocab.json";
		NSString *mergesName = [spec[@"merges"] isKindOfClass:NSString.class] ? spec[@"merges"] : @"merges.txt";
		NSString *pretokenization = [spec[@"pretokenizer"] isKindOfClass:NSString.class] ? spec[@"pretokenizer"] : nil;
		return [[NFKByteLevelBPETokenizer alloc] initWithVocabURL:[directory URLByAppendingPathComponent:vocabName]
													   mergesURL:[directory URLByAppendingPathComponent:mergesName]
												   specialTokens:specialTokens
												 pretokenization:pretokenization
															 eos:eos
															 bos:bos
														   error:outError];
	}
	if ([type isEqualToString:@"unigram"] || [type isEqualToString:@"sentencepiece"]) {
		NSString *vocabName = [spec[@"vocab"] isKindOfClass:NSString.class] ? spec[@"vocab"] : @"unigram.json";
		return [[NFKUnigramTokenizer alloc] initWithVocabURL:[directory URLByAppendingPathComponent:vocabName]
											   specialTokens:specialTokens
														 eos:eos
														 bos:bos
													   error:outError];
	}
	if ([type isEqualToString:@"wordpiece"]) {
		NSString *vocabName = [spec[@"vocab"] isKindOfClass:NSString.class] ? spec[@"vocab"] : @"wordpiece.json";
		return [[NFKWordPieceTokenizer alloc] initWithVocabURL:[directory URLByAppendingPathComponent:vocabName]
												 specialTokens:specialTokens
														   eos:eos
														   bos:bos
														 error:outError];
	}

	[self setError:outError
			  code:kNFKError_InferenceUnsupported
			reason:[NSString stringWithFormat:@"unsupported tokenizer type '%@'", type ?: @"(none)"]];
	return nil;
}

// A missing id is -1, not 0: 0 is a valid token in most vocabularies.
+ (NSInteger)tokenId:(nullable id)value
{
	return [value isKindOfClass:NSNumber.class] ? [value integerValue] : -1;
}

- (instancetype)initWithEOS:(NSInteger)eosTokenId bos:(NSInteger)bosTokenId
{
	self = [super init];
	if (self != nil) {
		_eosTokenId = eosTokenId;
		_bosTokenId = bosTokenId;
	}
	return self;
}

- (NSArray<NSString *> *)segmentsOfText:(NSString *)text
						bySpecialTokens:(nullable NSDictionary<NSString *, NSNumber *> *)specialTokens
{
	if (specialTokens.count == 0) {
		return @[text];
	}
	NSArray<NSString *> *literals = [specialTokens.allKeys sortedArrayUsingComparator:^NSComparisonResult(NSString *a, NSString *b) {
		return [@(b.length) compare:@(a.length)];
	}];
	NSMutableArray<NSString *> *escaped = [NSMutableArray arrayWithCapacity:literals.count];
	for (NSString *literal in literals) {
		[escaped addObject:[NSRegularExpression escapedPatternForString:literal]];
	}
	NSRegularExpression *expression =
		[NSRegularExpression regularExpressionWithPattern:[escaped componentsJoinedByString:@"|"] options:0 error:NULL];
	if (expression == nil) {
		return @[text];
	}

	NSMutableArray<NSString *> *segments = [NSMutableArray array];
	__block NSUInteger cursor = 0;
	[expression enumerateMatchesInString:text options:0 range:NSMakeRange(0, text.length)
							  usingBlock:^(NSTextCheckingResult *match, NSMatchingFlags flags, BOOL *stop) {
		if (match.range.location > cursor) {
			[segments addObject:[text substringWithRange:NSMakeRange(cursor, match.range.location - cursor)]];
		}
		[segments addObject:[text substringWithRange:match.range]];
		cursor = match.range.location + match.range.length;
	}];
	if (cursor < text.length) {
		[segments addObject:[text substringFromIndex:cursor]];
	}
	return segments;
}

- (NSArray<NSNumber *> *)encode:(NSString *)text
{
	[self doesNotRecognizeSelector:_cmd];
	return @[];
}

- (NSString *)decode:(NSArray<NSNumber *> *)tokenIds
{
	[self doesNotRecognizeSelector:_cmd];
	return @"";
}

+ (BOOL)setError:(NSError * _Nullable *)outError code:(NSInteger)code reason:(NSString *)reason
{
	if (outError != NULL) {
		*outError = [NSError errorWithDomain:NFKInferenceErrorDomain
										code:code
									userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return NO;
}

@end
