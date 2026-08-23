//
//  NFKCLIPTokenizer.m
//  InferKit
//

#import "NFKCLIPTokenizer.h"

// CLIP pre-tokenization: the two markers, then contractions, a run of letters, a SINGLE digit, or a
// run of other non-space characters. A digit at a time is what the reference does, so "2024" is four
// pieces.
static NSString * const kNFKCLIPPattern =
	@"<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+";

static NSString * const kNFKCLIPWordEnd = @"</w>";

@implementation NFKCLIPTokenizer
{
	NSRegularExpression *_whitespace;
}

- (NSString *)pretokenizationPattern
{
	return kNFKCLIPPattern;
}

- (NSString *)normalizedText:(NSString *)text
{
	if (_whitespace == nil) {
		_whitespace = [NSRegularExpression regularExpressionWithPattern:@"\\s+" options:0 error:NULL];
	}
	NSString *collapsed = [_whitespace stringByReplacingMatchesInString:text
																options:0
																  range:NSMakeRange(0, text.length)
														   withTemplate:@" "];
	return [[collapsed stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet] lowercaseString];
}

- (NSArray<NSString *> *)symbolsForWord:(NSString *)word
{
	NSMutableArray<NSString *> *symbols = [[super symbolsForWord:word] mutableCopy];
	if (symbols.count == 0) {
		return symbols;
	}
	symbols[symbols.count - 1] = [symbols.lastObject stringByAppendingString:kNFKCLIPWordEnd];
	return symbols;
}

- (NSString *)finalizedText:(NSString *)text
{
	NSString *spaced = [text stringByReplacingOccurrencesOfString:kNFKCLIPWordEnd withString:@" "];
	return [spaced stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
}

@end
